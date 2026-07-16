"""CommandHistory: the editor's undo/redo machinery (step-20).

Absorbs VisualScenarioEditor's undo_stack/redo_stack/max_undo_history and the
_history_position/_saved_history_position edit counters. execute/undo/redo
bodies are verbatim from the editor (prints included — they are part of the
observable CLI behavior), with exactly two substitutions:

- the editor's `_mark_scene_dirty()` call became the position bump here; the
  dirty-hint half now arrives via the on_change callback;
- the editor's `_refresh_active_info_panel()` calls became on_change
  notifications, fired at the same points (successful execute/undo/redo only
  — never on the failed-redo re-push path).

on_change(command, action) with action in {"execute", "undo", "redo"} is the
single wiring point back into the editor (dirty-hint + info-panel refresh);
per the refactor plan this callback and SessionState are the entire
mechanism budget — no event bus.
"""


class CommandHistory:
    def __init__(self, max_undo_history=25, on_change=None):
        self.undo_stack = []       # Commands that can be undone (max 25)
        self.redo_stack = []       # Commands that can be redone
        self.max_undo_history = max_undo_history
        self._on_change = on_change
        self._position = 0         # Monotonic counter of applied edits
        self._saved_position = 0   # Counter at last save/load

    # -- edit counters ------------------------------------------------------

    @property
    def position(self):
        return self._position

    @property
    def saved_position(self):
        return self._saved_position

    def mark_saved(self):
        """Record that the current edit position matches the saved file."""
        self._saved_position = self._position

    def reset_positions(self):
        """Zero both counters WITHOUT touching the stacks (scenario-load path)."""
        self._position = 0
        self._saved_position = 0

    def clear(self):
        """Drop both stacks and zero the counters (discard/reset/cleanup paths)."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._position = 0
        self._saved_position = 0

    # -- history operations (bodies verbatim from VisualScenarioEditor) -----

    def execute(self, command):
        """Execute a command and add it to the undo stack"""
        # Execute the command
        success = command.execute()
        if success is not False:  # Allow None or True
            self._position += 1
            # Add to undo stack
            self.undo_stack.append(command)
            # Clear redo stack since we have a new action
            if self.redo_stack:
                print(f"Clearing {len(self.redo_stack)} redo commands due to new action")
                self.redo_stack.clear()
            # Limit undo stack size
            if len(self.undo_stack) > self.max_undo_history:
                removed_command = self.undo_stack.pop(0)
                print(f"Removed oldest command from history: {removed_command.get_description()}")
            print(f"Executed: {command.get_description()} (Undo stack: {len(self.undo_stack)}, Redo stack: {len(self.redo_stack)})")
            self._notify(command, "execute")
        return success

    def undo(self):
        """Undo the last command (Ctrl+Z)"""
        if self.undo_stack:
            command = self.undo_stack.pop()
            command.undo()
            self.redo_stack.append(command)
            self._position = max(0, self._position - 1)
            # Limit redo stack size
            if len(self.redo_stack) > self.max_undo_history:
                self.redo_stack.pop(0)
            print(f"Undid: {command.get_description()} (Undo stack: {len(self.undo_stack)}, Redo stack: {len(self.redo_stack)})")
            self._notify(command, "undo")
            return True
        else:
            print("Nothing to undo")
            return False

    def redo(self):
        """Redo the last undone command (Ctrl+Y)"""
        if self.redo_stack:
            command = self.redo_stack.pop()
            # Use redo() method if available, otherwise fall back to execute()
            if hasattr(command, 'redo'):
                success = command.redo()
            else:
                success = command.execute()
            if success is not False:
                self.undo_stack.append(command)
                self._position += 1
                print(f"Redid: {command.get_description()} (Undo stack: {len(self.undo_stack)}, Redo stack: {len(self.redo_stack)})")
                self._notify(command, "redo")
                return True
            else:
                # If redo failed, put command back in redo stack
                self.redo_stack.append(command)
                print(f"Failed to redo: {command.get_description()}")
                return False
        else:
            print("Nothing to redo")
            return False

    # -- maintenance --------------------------------------------------------

    def remap_vehicle_id(self, old_id, new_id, new_actor=None):
        """Rewrite command targets after a respawn changed an actor id.

        Body verbatim from CameraImageProcessor.update_vehicle_id_references'
        stack loop — its one reach into the editor's stacks. Actor references
        are updated when the replacement object is available.
        """
        for command in self.undo_stack + self.redo_stack:
            if hasattr(command, 'vehicle_id') and command.vehicle_id == old_id:
                command.vehicle_id = new_id
            if hasattr(command, 'current_vehicle_id') and command.current_vehicle_id == old_id:
                command.current_vehicle_id = new_id
            # Personal-trigger commands (Set/Move) keep their owner id inside a
            # selection dict (fix-21): without this rewrite, undoing a trigger
            # delete/move after the owner was respawned targets a dead id — the
            # trigger is silently lost while the console still prints "Undid".
            selection = getattr(command, 'selection', None)
            if (isinstance(selection, dict)
                    and selection.get('kind') in ('vehicle', 'pedestrian')
                    and selection.get('id') == old_id):
                selection['id'] = new_id
            if new_actor is not None:
                for attribute in ('spawned_vehicle', 'vehicle'):
                    actor = getattr(command, attribute, None)
                    if actor is not None and getattr(actor, 'id', None) == old_id:
                        setattr(command, attribute, new_actor)

    def _notify(self, command, action):
        if self._on_change is not None:
            self._on_change(command, action)
