from ai.guard_agent import GuardAgent

class StealthSystem:
    @staticmethod
    def calculate_visibility(guard_pos, player_pos, grid):
        return GuardAgent.line_of_sight(guard_pos, player_pos, grid)
    
    @staticmethod
    def update_last_known(visible, player_pos, last_known):
        return player_pos if visible else last_known