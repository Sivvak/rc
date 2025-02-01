import numpy as np

class BayesFilter:
    def __init__(self):
        self.grid_size = 10
        # Initialize belief state - initially agent is at (0,0) with probability 1
        self.belief = np.zeros((10, 10))
        self.belief[0, 0] = 1.0
        
    def get_cell_color(self, row, col):
        """Returns true color of the cell"""
        return 'R' if row % 2 == 0 else 'G'
    
    def is_valid_position(self, row, col):
        """Check if position is within grid bounds"""
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size
    
    def motion_update(self, command):
        """Update belief state based on motion model"""
        new_belief = np.zeros((10, 10))
        
        # For each cell in the grid
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.belief[row, col] > 0:
                    # Calculate new position if command is followed
                    new_row, new_col = row, col
                    if command == 'u':
                        new_row = row - 1
                    elif command == 'd':
                        new_row = row + 1
                    elif command == 'r':
                        new_col = col + 1
                    elif command == 'l':
                        new_col = col - 1
                    
                    # If move is valid, agent moves with 0.8 probability
                    if self.is_valid_position(new_row, new_col):
                        new_belief[new_row, new_col] += 0.8 * self.belief[row, col]
                        new_belief[row, col] += 0.2 * self.belief[row, col]
                    else:
                        # If move is invalid, agent stays in place with probability 1
                        new_belief[row, col] += self.belief[row, col]
        
        self.belief = new_belief
        
    def sensor_update(self, observation):
        """Update belief state based on sensor model"""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                true_color = self.get_cell_color(row, col)
                # Sensor probability calculation
                sensor_prob = 0.75 if observation == true_color else 0.25
                self.belief[row, col] *= sensor_prob
        
        # Normalize beliefs
        if np.sum(self.belief) > 0:
            self.belief /= np.sum(self.belief)
    
    def process_input(self, input_string):
        """Process sequence of commands and observations"""
        for char in input_string:
            if char in 'udrl':  # Motion commands
                self.motion_update(char)
            elif char in 'RG':  # Sensor observations
                self.sensor_update(char)
        
        return self.belief

def main():
    # Example usage
    filter = BayesFilter()
    input_sequence = input("Enter sequence of commands and observations: ")
    result = filter.process_input(input_sequence)
    print("\nFinal belief state:")
    print(result)
    
if __name__ == "__main__":
    main() 