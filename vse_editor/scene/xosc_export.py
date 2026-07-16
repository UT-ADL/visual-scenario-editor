"""OpenSCENARIO (.xosc) export (moved verbatim from CameraImageProcessor,
self -> processor rename only — step-22).

A one-way exporter: builds the same in-memory snapshot as the JSON save
(scenario_io._collect_scenario_snapshot via the processor delegate) and
emits OpenSCENARIO XML plus the _tl_freeze.py helper next to the output.
Deliberately independent of the JSON writer (parallel serializer for a
different format — do not merge them).
"""

import os
import traceback

from vse_common.traffic_lights import TRAFFIC_LIGHT_FINGERPRINT_SCALE
from vse_editor.scene.scenario_io import _weather_keyframes_for_save


def export_to_openscenario(processor, filename):
    """Export current scenario to OpenSCENARIO 1.0 .xosc format.

    Preserved: entity spawns, waypoint paths, speed, per-waypoint idle times,
               proximity triggers, traffic-light triggers (pos-based),
               weather (first keyframe, static), map name, NPCs, vehicle color.
    Lost:      speed_deviation, animated weather,
               ignore_* flags, max_lat_acc.
    """
    import xml.etree.ElementTree as ET
    import math

    # ── helpers ──────────────────────────────────────────────────────────

    def _fmt(v):
        return f"{float(v):.6f}"

    def _deg2rad(d):
        return float(d) * math.pi / 180.0

    def _world_pos(parent, x, y, z, h=0.0, p=0.0, r=0.0):
        wp = ET.SubElement(parent, 'WorldPosition')
        wp.set('x', _fmt(x)); wp.set('y', _fmt(y)); wp.set('z', _fmt(z))
        wp.set('h', _fmt(h)); wp.set('p', _fmt(p)); wp.set('r', _fmt(r))
        return wp

    def _vehicle_category(type_str):
        t = (type_str or '').lower()
        if any(k in t for k in ('crossbike', 'bike', 'cycle', 'vespa', 'bh.')):
            return 'bicycle'
        if any(k in t for k in ('truck', 'cola', 'firetruck')):
            return 'truck'
        if any(k in t for k in ('van', '.t2', 'ambulance')):
            return 'van'
        if any(k in t for k in ('motorcycle', 'moto', 'kawasaki', 'yamaha', 'harley')):
            return 'motorbike'
        return 'car'

    def _cloud_state(cloudiness):
        c = float(cloudiness or 0)
        if c < 10:   return 'free'
        if c < 40:   return 'cloudy'
        if c < 70:   return 'overcast'
        return 'rainy'

    def _add_bbox(parent, is_ped=False):
        bb = ET.SubElement(parent, 'BoundingBox')
        c = ET.SubElement(bb, 'Center')
        c.set('x', '1.5'); c.set('y', '0.0'); c.set('z', '0.9')
        d = ET.SubElement(bb, 'Dimensions')
        d.set('width', '2.1' if is_ped else '2.0')
        d.set('length', '4.5'); d.set('height', '1.8')

    def _add_vehicle_entity(entities, name, v_data, is_ego=False):
        so = ET.SubElement(entities, 'ScenarioObject')
        so.set('name', name)
        type_str = v_data.get('type', 'vehicle.tesla.model3')
        veh = ET.SubElement(so, 'Vehicle')
        veh.set('name', type_str)
        veh.set('vehicleCategory', _vehicle_category(type_str))
        ET.SubElement(veh, 'ParameterDeclarations')
        perf = ET.SubElement(veh, 'Performance')
        perf.set('maxSpeed', '69.444')
        perf.set('maxAcceleration', '10.0')
        perf.set('maxDeceleration', '10.0')
        _add_bbox(veh, is_ped=False)
        axles = ET.SubElement(veh, 'Axles')
        fa = ET.SubElement(axles, 'FrontAxle')
        fa.set('maxSteering', '0.5'); fa.set('wheelDiameter', '0.6')
        fa.set('trackWidth', '1.8'); fa.set('positionX', '3.1'); fa.set('positionZ', '0.3')
        ra = ET.SubElement(axles, 'RearAxle')
        ra.set('maxSteering', '0.0'); ra.set('wheelDiameter', '0.6')
        ra.set('trackWidth', '1.8'); ra.set('positionX', '0.0'); ra.set('positionZ', '0.3')
        props = ET.SubElement(veh, 'Properties')
        pt = ET.SubElement(props, 'Property')
        pt.set('name', 'type')
        pt.set('value', 'ego_vehicle' if is_ego else 'simulation')
        color = v_data.get('color') or ('0,0,255' if is_ego else None)
        if color:
            pc = ET.SubElement(props, 'Property')
            pc.set('name', 'color'); pc.set('value', str(color))

    def _add_pedestrian_entity(entities, name, v_data):
        so = ET.SubElement(entities, 'ScenarioObject')
        so.set('name', name)
        type_str = v_data.get('type', 'walker.pedestrian.0001')
        ped = ET.SubElement(so, 'Pedestrian')
        ped.set('model', type_str); ped.set('mass', '90.0')
        ped.set('name', type_str); ped.set('pedestrianCategory', 'pedestrian')
        ET.SubElement(ped, 'ParameterDeclarations')
        _add_bbox(ped, is_ped=True)
        props = ET.SubElement(ped, 'Properties')
        pt = ET.SubElement(props, 'Property')
        pt.set('name', 'type'); pt.set('value', 'simulation')

    def _private_teleport(actions_elem, entity_name, loc, rot):
        priv = ET.SubElement(actions_elem, 'Private')
        priv.set('entityRef', entity_name)
        pa = ET.SubElement(priv, 'PrivateAction')
        ta = ET.SubElement(pa, 'TeleportAction')
        pos = ET.SubElement(ta, 'Position')
        _world_pos(pos, loc['x'], loc['y'], loc['z'],
                   h=_deg2rad(rot.get('yaw', 0.0)),
                   p=_deg2rad(rot.get('pitch', 0.0)),
                   r=_deg2rad(rot.get('roll', 0.0)))
        return priv

    def _add_ego_controller(priv_elem):
        pa = ET.SubElement(priv_elem, 'PrivateAction')
        ca = ET.SubElement(pa, 'ControllerAction')
        aca = ET.SubElement(ca, 'AssignControllerAction')
        ctrl = ET.SubElement(aca, 'Controller')
        ctrl.set('name', 'EgoVehicleAgent')
        props = ET.SubElement(ctrl, 'Properties')
        pm = ET.SubElement(props, 'Property')
        pm.set('name', 'module'); pm.set('value', 'external_control')
        ocva = ET.SubElement(ca, 'OverrideControllerValueAction')
        for tag in ('Throttle', 'Brake', 'Clutch', 'ParkingBrake', 'SteeringWheel'):
            e = ET.SubElement(ocva, tag)
            e.set('value', '0'); e.set('active', 'false')
        gear = ET.SubElement(ocva, 'Gear')
        gear.set('number', '0'); gear.set('active', 'false')

    def _add_npc_controller(priv_elem, npc_name, v_data):
        pa = ET.SubElement(priv_elem, 'PrivateAction')
        ca = ET.SubElement(pa, 'ControllerAction')
        aca = ET.SubElement(ca, 'AssignControllerAction')
        ctrl = ET.SubElement(aca, 'Controller')
        ctrl.set('name', f'{npc_name}Controller')
        props = ET.SubElement(ctrl, 'Properties')
        pm = ET.SubElement(props, 'Property')
        pm.set('name', 'module'); pm.set('value', 'simple_vehicle_control')
        ptl = ET.SubElement(props, 'Property')
        ptl.set('name', 'consider_trafficlights')
        ptl.set('value', 'false' if v_data.get('ignore_traffic_lights') else 'true')
        # Note: consider_obstacles is omitted — it spawns a sensor.other.obstacle
        # per vehicle, which kills FPS with many NPCs.
        ocva = ET.SubElement(ca, 'OverrideControllerValueAction')
        for tag in ('Throttle', 'Brake', 'Clutch', 'ParkingBrake', 'SteeringWheel'):
            e = ET.SubElement(ocva, tag)
            e.set('value', '0'); e.set('active', 'false')
        gear = ET.SubElement(ocva, 'Gear')
        gear.set('number', '0'); gear.set('active', 'false')

    def _polyline_vertices(spawn_loc, spawn_rot, waypoints, default_speed_kmh):
        """Return list of (x, y, z, h_rad, time_s) for all trajectory vertices."""
        verts = []
        px = spawn_loc['x']; py = spawn_loc['y']; pz = spawn_loc['z']
        h0 = _deg2rad(spawn_rot.get('yaw', 0.0))
        verts.append((px, py, pz, h0, 0.0))
        t = 0.0
        for wp in (waypoints or []):
            loc = wp.get('location', {})
            x = loc.get('x', px); y = loc.get('y', py); z = loc.get('z', pz)
            speed_ms = max((wp.get('speed_km_h') or default_speed_kmh or 5.0) / 3.6, 0.001)
            dist = math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
            t += dist / speed_ms
            # heading: direction of travel, or explicit yaw if provided
            explicit_yaw = wp.get('yaw')
            if explicit_yaw is not None:
                h = _deg2rad(explicit_yaw)
            elif dist > 0.01:
                h = math.atan2(y - py, x - px)
            else:
                h = verts[-1][3]
            verts.append((x, y, z, h, t))
            idle = float(wp.get('idle_time_s') or 0.0)
            if idle > 0:
                t += idle
                verts.append((x, y, z, h, t))   # duplicate vertex = pause
            px, py, pz = x, y, z
        return verts


    def _build_speed_segments(spawn_loc, waypoints, default_speed_kmh):
        """Build SpeedAction-based segments (same pattern as PedestrianCrossingFront.xosc).
        Returns (segments, init_heading_rad).
        Segment types:
          ('walk', distance_m, speed_ms)       — SpeedAction dynamicsDimension="distance"
          ('idle', duration_s)                  — SpeedAction dynamicsDimension="time" speed=0
          ('teleport', x, y, z, h_rad)          — TeleportAction to reposition + reorient
        init_heading_rad: heading from spawn toward first waypoint.
        """
        segments = []
        px, py, pz = spawn_loc['x'], spawn_loc['y'], spawn_loc['z']
        # Departure-speed semantics (match vse_play.py playback): each leg is
        # walked at the speed of the point it leaves. The first leg leaves the
        # spawn at the pedestrian's initial speed; later legs carry the previous
        # waypoint's speed. The final waypoint's speed is never used to travel.
        departure_speed_kmh = default_speed_kmh
        for i, wp in enumerate(waypoints):
            loc = wp.get('location', {})
            x = loc.get('x', px); y = loc.get('y', py); z = loc.get('z', pz)
            speed_ms = max((departure_speed_kmh or 5.0) / 3.6, 0.001)
            dist = math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
            if dist > 0.01:
                segments.append(('walk', dist, speed_ms))
            # This waypoint's speed governs the next leg.
            departure_speed_kmh = wp.get('speed_km_h') or default_speed_kmh
            idle = float(wp.get('idle_time_s') or 0.0)
            if idle > 0:
                segments.append(('idle', idle))
            # Brief pause + teleport + settling pause at each intermediate waypoint
            if i < len(waypoints) - 1:
                segments.append(('idle', 0.1))   # zero speed before teleport
                nxt = waypoints[i + 1].get('location', {})
                h = math.atan2(nxt.get('y', y) - y, nxt.get('x', x) - x)
                segments.append(('teleport', x, y, z + 0.5, h))  # +0.5m to stay above ground
                segments.append(('idle', 0.3))   # let walker rotation settle after teleport
            px, py, pz = x, y, z
        first_loc = waypoints[0].get('location', {})
        init_h = math.atan2(first_loc.get('y', py) - spawn_loc['y'],
                            first_loc.get('x', px) - spawn_loc['x'])
        return segments, init_h

    def _add_chained_start_trigger(parent, prev_action_ref):
        """Add a StartTrigger that fires when prev_action_ref reaches completeState."""
        st = ET.SubElement(parent, 'StartTrigger')
        cg = ET.SubElement(st, 'ConditionGroup')
        cond = ET.SubElement(cg, 'Condition')
        cond.set('name', f'After_{prev_action_ref}')
        cond.set('delay', '0'); cond.set('conditionEdge', 'rising')
        bvc = ET.SubElement(cond, 'ByValueCondition')
        sbesc = ET.SubElement(bvc, 'StoryboardElementStateCondition')
        sbesc.set('storyboardElementType', 'action')
        sbesc.set('storyboardElementRef', prev_action_ref)
        sbesc.set('state', 'completeState')

    def _add_reach_position_trigger(parent, entity_ref, x, y, z, tolerance=5.0):
        """Add a StartTrigger that fires when entity_ref reaches a WorldPosition."""
        st = ET.SubElement(parent, 'StartTrigger')
        cg = ET.SubElement(st, 'ConditionGroup')
        cond = ET.SubElement(cg, 'Condition')
        cond.set('name', 'AtPosition'); cond.set('delay', '0')
        cond.set('conditionEdge', 'rising')
        bec = ET.SubElement(cond, 'ByEntityCondition')
        te = ET.SubElement(bec, 'TriggeringEntities')
        te.set('triggeringEntitiesRule', 'any')
        ET.SubElement(te, 'EntityRef').set('entityRef', entity_ref)
        ec = ET.SubElement(bec, 'EntityCondition')
        rpc = ET.SubElement(ec, 'ReachPositionCondition')
        rpc.set('tolerance', _fmt(tolerance))
        pos = ET.SubElement(rpc, 'Position')
        _world_pos(pos, x, y, z)

    def _tl_freeze_action(event_elem, action_name, x, y, seq_str):
        """Add a UserDefinedAction that runs _tl_freeze.py to freeze a traffic light.

        seq_str is a pre-built sequence like 'red:10.0,green' (see _build_tl_seq).
        """
        ae = ET.SubElement(event_elem, 'Action')
        ae.set('name', action_name)
        uda = ET.SubElement(ae, 'UserDefinedAction')
        cca = ET.SubElement(uda, 'CustomCommandAction')
        cca.set('type',
                 f'python3 _tl_freeze.py {_fmt(x)} {_fmt(y)} {seq_str}')

    def _add_start_trigger(parent, v_data, ego_ref,
                           global_trigger=None, has_ego=False):
        st = ET.SubElement(parent, 'StartTrigger')
        trigger = v_data.get('trigger')
        idle_s = float(v_data.get('idle_time_s') or 0.0)
        cg = ET.SubElement(st, 'ConditionGroup')
        cond = ET.SubElement(cg, 'Condition')
        cond.set('name', 'StartCondition')
        cond.set('conditionEdge', 'rising')
        if trigger:
            # Personal trigger → ego must reach it
            cond.set('delay', _fmt(idle_s))
            bec = ET.SubElement(cond, 'ByEntityCondition')
            te = ET.SubElement(bec, 'TriggeringEntities')
            te.set('triggeringEntitiesRule', 'any')
            ET.SubElement(te, 'EntityRef').set('entityRef', ego_ref)
            ec = ET.SubElement(bec, 'EntityCondition')
            rpc = ET.SubElement(ec, 'ReachPositionCondition')
            rpc.set('tolerance', _fmt(trigger.get('radius', 10.0)))
            c = trigger.get('center', {})
            _world_pos(ET.SubElement(rpc, 'Position'),
                       c.get('x', 0), c.get('y', 0), c.get('z', 0))
        elif has_ego and global_trigger:
            # No personal trigger, but global trigger exists → ego must reach global trigger
            cond.set('delay', _fmt(idle_s))
            loc = global_trigger.get('location', {})
            bec = ET.SubElement(cond, 'ByEntityCondition')
            te = ET.SubElement(bec, 'TriggeringEntities')
            te.set('triggeringEntitiesRule', 'any')
            ET.SubElement(te, 'EntityRef').set('entityRef', ego_ref)
            ec = ET.SubElement(bec, 'EntityCondition')
            rpc = ET.SubElement(ec, 'ReachPositionCondition')
            rpc.set('tolerance', _fmt(global_trigger.get('radius', 10.0)))
            _world_pos(ET.SubElement(rpc, 'Position'),
                       loc.get('x', 0), loc.get('y', 0), loc.get('z', 0))
        elif has_ego:
            # Ego present but no trigger at all → NPC never activates
            cond.set('delay', '0')
            bvc = ET.SubElement(cond, 'ByValueCondition')
            stc = ET.SubElement(bvc, 'SimulationTimeCondition')
            stc.set('value', '999999'); stc.set('rule', 'greaterThan')
        else:
            # No ego → start immediately after idle time
            cond.set('delay', '0')
            bvc = ET.SubElement(cond, 'ByValueCondition')
            stc = ET.SubElement(bvc, 'SimulationTimeCondition')
            stc.set('value', _fmt(idle_s)); stc.set('rule', 'greaterThan')

    def _add_environment_action(actions_elem, wf):
        ga = ET.SubElement(actions_elem, 'GlobalAction')
        ea = ET.SubElement(ga, 'EnvironmentAction')
        env = ET.SubElement(ea, 'Environment')
        env.set('name', 'Environment1')
        tod = ET.SubElement(env, 'TimeOfDay')
        tod.set('animation', 'false'); tod.set('dateTime', '2020-03-24T12:00:00')
        weather_elem = ET.SubElement(env, 'Weather')
        weather_elem.set('cloudState', _cloud_state(wf.get('cloudiness', 0)))
        sun = ET.SubElement(weather_elem, 'Sun')
        alt_rad = _deg2rad(wf.get('sun_altitude_angle', 45.0))
        sun.set('intensity', _fmt(max(0.0, min(1.0, math.sin(alt_rad)))))
        sun.set('azimuth', _fmt(_deg2rad(wf.get('sun_azimuth_angle', 0.0))))
        sun.set('elevation', _fmt(alt_rad))
        fog_elem = ET.SubElement(weather_elem, 'Fog')
        fog_d = float(wf.get('fog_density') or 0.0)
        fog_elem.set('visualRange', _fmt(100000.0 * (1.0 - fog_d / 100.0)))
        prec = ET.SubElement(weather_elem, 'Precipitation')
        prec_val = float(wf.get('precipitation') or 0.0)
        prec.set('precipitationType', 'rain' if prec_val > 0 else 'dry')
        prec.set('intensity', _fmt(prec_val / 100.0))
        rc = ET.SubElement(env, 'RoadCondition')
        rc.set('frictionScaleFactor', '1.0')

    def _indent(elem, level=0):
        pad = '\n' + '  ' * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = pad + '  '
            for child in elem:
                _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = pad
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad

    # ── main body ────────────────────────────────────────────────────────
    try:
        map_obj = processor._get_cached_map(refresh=False)
        if map_obj:
            full_name = map_obj.name
            map_name = full_name.split('/')[-1] if '/' in full_name else full_name
        else:
            map_name = 'unknown'

        weather_kfs = _weather_keyframes_for_save(processor)
        save_data, _ = processor._collect_scenario_snapshot(map_name)
        save_data['weather_keyframes'] = weather_kfs

        vehicles = save_data.get('vehicles', [])
        ego = save_data.get('ego_vehicle')
        ego_ref = 'ego_vehicle'

        # ── root ─────────────────────────────────────────────────────────
        root = ET.Element('OpenSCENARIO')
        fh = ET.SubElement(root, 'FileHeader')
        fh.set('revMajor', '1'); fh.set('revMinor', '0')
        fh.set('date', '2020-03-24T12:00:00')
        fh.set('description', f'CARLA:{map_name}'); fh.set('author', '')
        ET.SubElement(root, 'ParameterDeclarations')
        ET.SubElement(root, 'CatalogLocations')
        rn = ET.SubElement(root, 'RoadNetwork')
        lf = ET.SubElement(rn, 'LogicFile'); lf.set('filepath', map_name)
        sgf = ET.SubElement(rn, 'SceneGraphFile'); sgf.set('filepath', '')

        # ── entities ─────────────────────────────────────────────────────
        entities = ET.SubElement(root, 'Entities')
        if ego:
            _add_vehicle_entity(entities, ego_ref, ego, is_ego=True)
        npc_names = []
        for i, v in enumerate(vehicles):
            name = f'adversary_{i + 1}' if len(vehicles) > 1 else 'adversary'
            npc_names.append(name)
            if str(v.get('type', '')).startswith('walker.'):
                _add_pedestrian_entity(entities, name, v)
            else:
                _add_vehicle_entity(entities, name, v, is_ego=False)

        # ── storyboard ───────────────────────────────────────────────────
        sb = ET.SubElement(root, 'Storyboard')
        init = ET.SubElement(sb, 'Init')
        init_acts = ET.SubElement(init, 'Actions')

        if weather_kfs:
            _add_environment_action(init_acts, weather_kfs[0])
        if ego:
            priv = _private_teleport(init_acts, ego_ref, ego['location'], ego['rotation'])
            _add_ego_controller(priv)
        for npc_name, v in zip(npc_names, vehicles):
            wps = v.get('waypoints', [])
            is_ped = str(v.get('type', '')).startswith('walker.')
            if wps and is_ped:
                # Pedestrian SpeedAction path: spawn heading must face first waypoint
                first_loc = wps[0].get('location', {})
                dx = first_loc.get('x', 0) - v['location']['x']
                dy = first_loc.get('y', 0) - v['location']['y']
                h_rad = math.atan2(dy, dx)
                aimed_rot = dict(v['rotation'])
                aimed_rot['yaw'] = h_rad * 180.0 / math.pi
                _private_teleport(init_acts, npc_name, v['location'], aimed_rot)
            else:
                # Vehicle or no waypoints: use original rotation from JSON
                priv = _private_teleport(init_acts, npc_name, v['location'], v['rotation'])
                if not is_ped and processor.vehicle_control_mode == 'velocity':
                    _add_npc_controller(priv, npc_name, v)

        # story
        story = ET.SubElement(sb, 'Story'); story.set('name', 'MyStory')
        act = ET.SubElement(story, 'Act'); act.set('name', 'Behavior')

        global_trigger = save_data.get('trigger')
        has_ego = ego is not None

        for npc_name, v in zip(npc_names, vehicles):
            mg = ET.SubElement(act, 'ManeuverGroup')
            mg.set('maximumExecutionCount', '1')
            mg.set('name', f'{npc_name}Sequence')
            actors = ET.SubElement(mg, 'Actors')
            actors.set('selectTriggeringEntities', 'false')
            ET.SubElement(actors, 'EntityRef').set('entityRef', npc_name)
            maneuver = ET.SubElement(mg, 'Maneuver')
            maneuver.set('name', f'{npc_name}Maneuver')
            waypoints = v.get('waypoints', [])
            if not waypoints:
                # No waypoints: single static speed event
                event = ET.SubElement(maneuver, 'Event')
                event.set('name', f'{npc_name}Move'); event.set('priority', 'overwrite')
                action_elem = ET.SubElement(event, 'Action')
                action_elem.set('name', f'{npc_name}MoveAction')
                pa = ET.SubElement(action_elem, 'PrivateAction')
                la = ET.SubElement(pa, 'LongitudinalAction')
                sa = ET.SubElement(la, 'SpeedAction')
                sad = ET.SubElement(sa, 'SpeedActionDynamics')
                sad.set('dynamicsShape', 'step')
                sad.set('value', '0'); sad.set('dynamicsDimension', 'time')
                sat = ET.SubElement(sa, 'SpeedActionTarget')
                ats = ET.SubElement(sat, 'AbsoluteTargetSpeed')
                ats.set('value', _fmt(v.get('speed_km_h', 0.0) / 3.6))
                _add_start_trigger(event, v, ego_ref,
                                   global_trigger=global_trigger,
                                   has_ego=has_ego)
                continue

            is_pedestrian = str(v.get('type', '')).startswith('walker.')

            if is_pedestrian:
                # ── Pedestrian: SpeedAction chain (PedestrianCrossingFront.xosc pattern) ──
                segments, _ = _build_speed_segments(
                    v['location'], waypoints, v.get('speed_km_h', 5.0))
                prev_action_ref = None
                is_first = True
                seg_counter = 0
                for seg in segments:
                    if seg[0] == 'walk':
                        distance_m, speed_ms = seg[1], seg[2]
                        event_name = f'{npc_name}Walk{seg_counter}'
                        action_name = f'{npc_name}Walk{seg_counter}Action'
                        ev = ET.SubElement(maneuver, 'Event')
                        ev.set('name', event_name); ev.set('priority', 'overwrite')
                        ae = ET.SubElement(ev, 'Action')
                        ae.set('name', action_name)
                        pa = ET.SubElement(ae, 'PrivateAction')
                        la = ET.SubElement(pa, 'LongitudinalAction')
                        sa = ET.SubElement(la, 'SpeedAction')
                        sad = ET.SubElement(sa, 'SpeedActionDynamics')
                        sad.set('dynamicsShape', 'step')
                        sad.set('value', _fmt(distance_m))
                        sad.set('dynamicsDimension', 'distance')
                        sat = ET.SubElement(sa, 'SpeedActionTarget')
                        ats = ET.SubElement(sat, 'AbsoluteTargetSpeed')
                        ats.set('value', _fmt(speed_ms))
                        if is_first:
                            _add_start_trigger(ev, v, ego_ref,
                                               global_trigger=global_trigger,
                                               has_ego=has_ego)
                            is_first = False
                        else:
                            _add_chained_start_trigger(ev, prev_action_ref)
                        prev_action_ref = action_name
                        seg_counter += 1

                    elif seg[0] == 'idle':
                        duration_s = seg[1]
                        event_name = f'{npc_name}Idle{seg_counter}'
                        action_name = f'{npc_name}Idle{seg_counter}Action'
                        ev = ET.SubElement(maneuver, 'Event')
                        ev.set('name', event_name); ev.set('priority', 'overwrite')
                        ae = ET.SubElement(ev, 'Action')
                        ae.set('name', action_name)
                        pa = ET.SubElement(ae, 'PrivateAction')
                        la = ET.SubElement(pa, 'LongitudinalAction')
                        sa = ET.SubElement(la, 'SpeedAction')
                        sad = ET.SubElement(sa, 'SpeedActionDynamics')
                        sad.set('dynamicsShape', 'step')
                        sad.set('value', _fmt(duration_s))
                        sad.set('dynamicsDimension', 'time')
                        sat = ET.SubElement(sa, 'SpeedActionTarget')
                        ats = ET.SubElement(sat, 'AbsoluteTargetSpeed')
                        ats.set('value', '0.0')
                        _add_chained_start_trigger(ev, prev_action_ref)
                        prev_action_ref = action_name
                        seg_counter += 1

                    elif seg[0] == 'teleport':
                        tx, ty, tz, th = seg[1], seg[2], seg[3], seg[4]
                        event_name = f'{npc_name}Repos{seg_counter}'
                        action_name = f'{npc_name}Repos{seg_counter}Action'
                        ev = ET.SubElement(maneuver, 'Event')
                        ev.set('name', event_name); ev.set('priority', 'overwrite')
                        ae = ET.SubElement(ev, 'Action')
                        ae.set('name', action_name)
                        pa = ET.SubElement(ae, 'PrivateAction')
                        ta = ET.SubElement(pa, 'TeleportAction')
                        pos = ET.SubElement(ta, 'Position')
                        _world_pos(pos, tx, ty, tz, h=th)
                        _add_chained_start_trigger(ev, prev_action_ref)
                        prev_action_ref = action_name
                        seg_counter += 1
                last_action_ref = prev_action_ref

                # Final stop event: set speed=0 so pedestrian doesn't walk forever
                stop_ev = ET.SubElement(maneuver, 'Event')
                stop_ev.set('name', f'{npc_name}Stop'); stop_ev.set('priority', 'overwrite')
                stop_ae = ET.SubElement(stop_ev, 'Action')
                stop_ae.set('name', f'{npc_name}StopAction')
                pa = ET.SubElement(stop_ae, 'PrivateAction')
                la = ET.SubElement(pa, 'LongitudinalAction')
                sa = ET.SubElement(la, 'SpeedAction')
                sad = ET.SubElement(sa, 'SpeedActionDynamics')
                sad.set('dynamicsShape', 'step')
                sad.set('value', '0'); sad.set('dynamicsDimension', 'time')
                sat = ET.SubElement(sa, 'SpeedActionTarget')
                ats = ET.SubElement(sat, 'AbsoluteTargetSpeed')
                ats.set('value', '0.0')
                _add_chained_start_trigger(stop_ev, last_action_ref)

            else:
                # ── Vehicle: AssignRouteAction + SpeedAction events ──
                # Event 1: Route assignment + initial speed
                first_speed_ms = max((waypoints[0].get('speed_km_h')
                                      or v.get('speed_km_h', 50.0)) / 3.6, 0.001)
                drive_ev = ET.SubElement(maneuver, 'Event')
                drive_ev.set('name', f'{npc_name}Drive')
                drive_ev.set('priority', 'overwrite')

                # Action A: AssignRouteAction with all waypoints
                route_ae = ET.SubElement(drive_ev, 'Action')
                route_ae.set('name', f'{npc_name}RouteAction')
                pa = ET.SubElement(route_ae, 'PrivateAction')
                ra = ET.SubElement(pa, 'RoutingAction')
                ara = ET.SubElement(ra, 'AssignRouteAction')
                route_el = ET.SubElement(ara, 'Route')
                route_el.set('name', f'{npc_name}Route')
                route_el.set('closed', 'false')
                for wp in waypoints:
                    wp_el = ET.SubElement(route_el, 'Waypoint')
                    # "shortest" bypasses GlobalRoutePlanner entirely —
                    # avoids wrong-lane snap and ego_next_wp crash.
                    # NpcVehicleControl still projects to road.
                    wp_el.set('routeStrategy', 'shortest')
                    loc = wp.get('location', {})
                    pos = ET.SubElement(wp_el, 'Position')
                    _world_pos(pos, loc.get('x', 0), loc.get('y', 0),
                               loc.get('z', 0))

                # Action B: Initial speed
                speed_ae = ET.SubElement(drive_ev, 'Action')
                speed_ae.set('name', f'{npc_name}InitSpeed')
                pa = ET.SubElement(speed_ae, 'PrivateAction')
                la = ET.SubElement(pa, 'LongitudinalAction')
                sa = ET.SubElement(la, 'SpeedAction')
                sad = ET.SubElement(sa, 'SpeedActionDynamics')
                sad.set('dynamicsShape', 'step')
                sad.set('value', '0'); sad.set('dynamicsDimension', 'time')
                sat = ET.SubElement(sa, 'SpeedActionTarget')
                ats = ET.SubElement(sat, 'AbsoluteTargetSpeed')
                ats.set('value', _fmt(first_speed_ms))

                _add_start_trigger(drive_ev, v, ego_ref,
                                   global_trigger=global_trigger,
                                   has_ego=has_ego)

                # Speed-change events (only where speed differs from previous)
                prev_speed_kmh = waypoints[0].get('speed_km_h') or v.get('speed_km_h', 50.0)
                speed_idx = 0
                for i, wp in enumerate(waypoints):
                    wp_speed_kmh = wp.get('speed_km_h') or v.get('speed_km_h', 50.0)
                    loc = wp.get('location', {})

                    # Idle event at this waypoint
                    idle_s = float(wp.get('idle_time_s') or 0.0)
                    if idle_s > 0:
                        idle_name = f'{npc_name}Idle{speed_idx}'
                        idle_action = f'{npc_name}Idle{speed_idx}Action'
                        ie = ET.SubElement(maneuver, 'Event')
                        ie.set('name', idle_name); ie.set('priority', 'parallel')
                        iae = ET.SubElement(ie, 'Action')
                        iae.set('name', idle_action)
                        pa = ET.SubElement(iae, 'PrivateAction')
                        la = ET.SubElement(pa, 'LongitudinalAction')
                        sa = ET.SubElement(la, 'SpeedAction')
                        sad = ET.SubElement(sa, 'SpeedActionDynamics')
                        sad.set('dynamicsShape', 'step')
                        sad.set('value', _fmt(idle_s))
                        sad.set('dynamicsDimension', 'time')
                        sat = ET.SubElement(sa, 'SpeedActionTarget')
                        ats = ET.SubElement(sat, 'AbsoluteTargetSpeed')
                        ats.set('value', '0.0')
                        _add_reach_position_trigger(
                            ie, npc_name, loc.get('x', 0), loc.get('y', 0),
                            loc.get('z', 0))
                        speed_idx += 1

                        # Resume event after idle
                        resume_speed = wp_speed_kmh
                        if i + 1 < len(waypoints):
                            resume_speed = (waypoints[i + 1].get('speed_km_h')
                                            or v.get('speed_km_h', 50.0))
                        resume_name = f'{npc_name}Resume{speed_idx}'
                        resume_action = f'{npc_name}Resume{speed_idx}Action'
                        re = ET.SubElement(maneuver, 'Event')
                        re.set('name', resume_name); re.set('priority', 'parallel')
                        rae = ET.SubElement(re, 'Action')
                        rae.set('name', resume_action)
                        pa = ET.SubElement(rae, 'PrivateAction')
                        la = ET.SubElement(pa, 'LongitudinalAction')
                        sa = ET.SubElement(la, 'SpeedAction')
                        sad = ET.SubElement(sa, 'SpeedActionDynamics')
                        sad.set('dynamicsShape', 'step')
                        sad.set('value', '0'); sad.set('dynamicsDimension', 'time')
                        sat = ET.SubElement(sa, 'SpeedActionTarget')
                        ats = ET.SubElement(sat, 'AbsoluteTargetSpeed')
                        ats.set('value', _fmt(resume_speed / 3.6))
                        _add_chained_start_trigger(re, idle_action)
                        speed_idx += 1
                        prev_speed_kmh = resume_speed
                        continue

                    # Speed change event (only if speed differs)
                    if i > 0 and abs(wp_speed_kmh - prev_speed_kmh) > 0.1:
                        sc_name = f'{npc_name}Speed{speed_idx}'
                        sc_action = f'{npc_name}Speed{speed_idx}Action'
                        se = ET.SubElement(maneuver, 'Event')
                        se.set('name', sc_name); se.set('priority', 'parallel')
                        sae = ET.SubElement(se, 'Action')
                        sae.set('name', sc_action)
                        pa = ET.SubElement(sae, 'PrivateAction')
                        la = ET.SubElement(pa, 'LongitudinalAction')
                        sa = ET.SubElement(la, 'SpeedAction')
                        sad = ET.SubElement(sa, 'SpeedActionDynamics')
                        sad.set('dynamicsShape', 'step')
                        sad.set('value', '0'); sad.set('dynamicsDimension', 'time')
                        sat = ET.SubElement(sa, 'SpeedActionTarget')
                        ats = ET.SubElement(sat, 'AbsoluteTargetSpeed')
                        ats.set('value', _fmt(wp_speed_kmh / 3.6))
                        _add_reach_position_trigger(
                            se, npc_name, loc.get('x', 0), loc.get('y', 0),
                            loc.get('z', 0))
                        speed_idx += 1
                    prev_speed_kmh = wp_speed_kmh

                # Final stop: speed=0 when route completes
                stop_ev = ET.SubElement(maneuver, 'Event')
                stop_ev.set('name', f'{npc_name}Stop')
                stop_ev.set('priority', 'parallel')
                stop_ae = ET.SubElement(stop_ev, 'Action')
                stop_ae.set('name', f'{npc_name}StopAction')
                pa = ET.SubElement(stop_ae, 'PrivateAction')
                la = ET.SubElement(pa, 'LongitudinalAction')
                sa = ET.SubElement(la, 'SpeedAction')
                sad = ET.SubElement(sa, 'SpeedActionDynamics')
                sad.set('dynamicsShape', 'step')
                sad.set('value', '0'); sad.set('dynamicsDimension', 'time')
                sat = ET.SubElement(sa, 'SpeedActionTarget')
                ats = ET.SubElement(sat, 'AbsoluteTargetSpeed')
                ats.set('value', '0.0')
                _add_chained_start_trigger(stop_ev, f'{npc_name}RouteAction')

        # ── traffic light triggers ────────────────────────────────────
        tl_triggers = save_data.get('traffic_light_triggers', [])
        tl_triggers_with_seq = [
            t for t in tl_triggers
            if t.get('sequence') and t.get('fingerprint')
        ]
        if tl_triggers_with_seq:
            # Write _tl_freeze.py helper alongside the xosc.
            # RunScript resolves paths relative to the xosc directory.
            _tl_helper_dir = os.path.dirname(os.path.abspath(filename))
            _tl_helper_path = os.path.join(_tl_helper_dir, '_tl_freeze.py')
            with open(_tl_helper_path, 'w') as _f:
                _f.write(
                    '#!/usr/bin/env python3\n'
                    '"""Freeze a CARLA traffic light by position.\n'
                    'Usage: python3 _tl_freeze.py <x> <y> <sequence>\n'
                    'Sequence format: color:duration,color:duration,...,color\n'
                    'Example: red:10,green:180  (red 10s then green forever)\n'
                    'Auto-generated by VSE OpenSCENARIO export.\n'
                    '"""\n'
                    'import sys, time, carla\n'
                    '\n'
                    'STATES = {"GREEN": carla.TrafficLightState.Green,\n'
                    '          "RED": carla.TrafficLightState.Red,\n'
                    '          "YELLOW": carla.TrafficLightState.Yellow,\n'
                    '          "OFF": carla.TrafficLightState.Off}\n'
                    '\n'
                    'x, y = float(sys.argv[1]), float(sys.argv[2])\n'
                    'steps = []\n'
                    'for part in sys.argv[3].split(","):\n'
                    '    tokens = part.split(":")\n'
                    '    color = STATES.get(tokens[0].upper())\n'
                    '    if color is None:\n'
                    '        sys.exit(f"Unknown color: {tokens[0]}")\n'
                    '    dur = float(tokens[1]) if len(tokens) > 1 else None\n'
                    '    steps.append((color, dur))\n'
                    '\n'
                    'client = carla.Client("localhost", 2000)\n'
                    'client.set_timeout(10.0)\n'
                    'world = client.get_world()\n'
                    'world.wait_for_tick()\n'
                    'tl = None\n'
                    'for actor in world.get_actors().filter("traffic.traffic_light"):\n'
                    '    loc = actor.get_transform().location\n'
                    '    if loc.distance(carla.Location(x, y, loc.z)) < 2.0:\n'
                    '        tl = actor\n'
                    '        break\n'
                    'if tl is None:\n'
                    '    sys.exit(f"No traffic light near ({x}, {y})")\n'
                    '\n'
                    'for color, dur in steps:\n'
                    '    tl.set_state(color)\n'
                    '    tl.set_green_time(99999)\n'
                    '    tl.set_red_time(99999)\n'
                    '    tl.set_yellow_time(99999)\n'
                    '    if dur is not None:\n'
                    '        time.sleep(dur)\n'
                )

            tl_mg = ET.SubElement(act, 'ManeuverGroup')
            tl_mg.set('name', 'trafficLightControl')
            tl_mg.set('maximumExecutionCount', '1')
            tl_actors = ET.SubElement(tl_mg, 'Actors')
            tl_actors.set('selectTriggeringEntities', 'false')
            ET.SubElement(tl_actors, 'EntityRef').set('entityRef', ego_ref)
            tl_man = ET.SubElement(tl_mg, 'Maneuver')
            tl_man.set('name', 'trafficLightManeuver')

            for ti, tlt in enumerate(tl_triggers_with_seq):
                center = tlt.get('center', {})
                radius = float(tlt.get('radius', 10.0))
                fingerprint = tlt.get('fingerprint', [])
                sequence = tlt.get('sequence', [])

                # Build sequence string: "red:5.0,green" (color:duration pairs)
                seq_parts = []
                for si, step in enumerate(sequence):
                    color = (step.get('color') or 'off').lower()
                    duration_ticks = float(step.get('duration_ticks', 100))
                    duration_s = duration_ticks * 0.05  # 20 Hz default
                    if si < len(sequence) - 1:
                        seq_parts.append(f'{color}:{duration_s:.1f}')
                    else:
                        seq_parts.append(color)  # last step: no duration
                seq_str = ','.join(seq_parts)

                # Single event per trigger (same pattern as vehicle triggers)
                ev = ET.SubElement(tl_man, 'Event')
                ev.set('name', f'tl{ti}_freeze')
                ev.set('priority', 'overwrite')

                # One action per traffic light in this trigger
                fp_scale = TRAFFIC_LIGHT_FINGERPRINT_SCALE
                for li, fp in enumerate(fingerprint):
                    _tl_freeze_action(
                        ev, f'tl{ti}_L{li}',
                        float(fp[0]) / fp_scale,
                        float(fp[1]) / fp_scale, seq_str)

                # StartTrigger: ego proximity (same as vehicle triggers)
                _add_reach_position_trigger(
                    ev, ego_ref,
                    center.get('x', 0),
                    center.get('y', 0),
                    center.get('z', 0),
                    tolerance=radius)

        # act start / stop triggers
        act_st = ET.SubElement(act, 'StartTrigger')
        act_st_cg = ET.SubElement(act_st, 'ConditionGroup')
        act_st_c = ET.SubElement(act_st_cg, 'Condition')
        act_st_c.set('name', 'OverallStartCondition')
        act_st_c.set('delay', '0'); act_st_c.set('conditionEdge', 'rising')
        act_st_bec = ET.SubElement(act_st_c, 'ByEntityCondition')
        act_st_te = ET.SubElement(act_st_bec, 'TriggeringEntities')
        act_st_te.set('triggeringEntitiesRule', 'any')
        ET.SubElement(act_st_te, 'EntityRef').set('entityRef', ego_ref)
        act_st_ec = ET.SubElement(act_st_bec, 'EntityCondition')
        ET.SubElement(act_st_ec, 'TraveledDistanceCondition').set('value', '1.0')

        act_stop = ET.SubElement(act, 'StopTrigger')

        # Act ends naturally when all NPC ManeuverGroups complete
        # (CARLA uses SUCCESS_ON_ALL for ManeuverGroups).
        # Ego destination is intentionally not used as a stop condition —
        # only NPC completion and the timeout fallback matter.

        # ConditionGroup: timeout fallback (1800 s)
        timeout_cg = ET.SubElement(act_stop, 'ConditionGroup')
        timeout_c = ET.SubElement(timeout_cg, 'Condition')
        timeout_c.set('name', 'Timeout')
        timeout_c.set('delay', '0'); timeout_c.set('conditionEdge', 'rising')
        timeout_bvc = ET.SubElement(timeout_c, 'ByValueCondition')
        timeout_stc = ET.SubElement(timeout_bvc, 'SimulationTimeCondition')
        timeout_stc.set('value', '1800'); timeout_stc.set('rule', 'greaterThan')

        # global stop trigger: standard CARLA evaluation criteria
        global_stop = ET.SubElement(sb, 'StopTrigger')
        for crit, param_ref, param_val in [
            ('criteria_CollisionTest',        '',                  ''),
            ('criteria_DrivenDistanceTest',   'distance_success',  '100'),
        ]:
            cg = ET.SubElement(global_stop, 'ConditionGroup')
            c = ET.SubElement(cg, 'Condition')
            c.set('name', crit); c.set('delay', '0'); c.set('conditionEdge', 'rising')
            bvc = ET.SubElement(c, 'ByValueCondition')
            pc = ET.SubElement(bvc, 'ParameterCondition')
            pc.set('parameterRef', param_ref)
            pc.set('value', param_val); pc.set('rule', 'lessThan')

        # ── write file ───────────────────────────────────────────────────
        _indent(root)
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0"?>\n')
            ET.ElementTree(root).write(f, encoding='unicode', xml_declaration=False)

        print(f"Exported OpenSCENARIO ({len(vehicles)} NPC(s)) to {filename}")
        return True

    except Exception as exc:
        print(f"Error exporting to OpenSCENARIO: {exc}")
        traceback.print_exc()
        return False
