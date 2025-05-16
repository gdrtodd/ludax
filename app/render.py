
import math
import svgwrite
from svgwrite import cm, mm

from config import BoardShapes, P1, P2, RENDER_CONFIG


class InteractiveBoardHandler():
    '''
    Facilitates an interactive version of a game environment by generating an SVG board that can be clicked on
    and mapping clicks to actions in the game environment
    '''
    def __init__(self,
                 game_info,
                 render_config=RENDER_CONFIG):
        
        self.game_info = game_info
        self.render_config = render_config
        self.cell_size = render_config['cell_size']
        self.padding = self.cell_size / 2
        

        if self.game_info.board_shape == BoardShapes.SQUARE or self.game_info.board_shape == BoardShapes.RECTANGLE:
            self.cell_size *= 2
            height, width = self.game_info.board_dims
            self.action_indices = [(y, x) for y in range(height) for x in range(width)]
            self.action_to_pixel = lambda action: self._grid_to_pixel(self.action_indices[action])
            self.pixel_to_action = lambda pixel: self.action_indices.index(self._pixel_to_grid(pixel))
            self.get_cell_vertices = lambda position: self._get_grid_vertices(position)

            # Compute the total width and height of the board
            self.total_width = width * self.cell_size
            self.total_height = height * self.cell_size
        
        elif self.game_info.board_shape == BoardShapes.HEXAGON:
            self.orientation = render_config['hexagon_orientation']

            # Compute the ordered sequence or (q, r, s) indices that correspond to horizontal scans over the board
            diameter = self.game_info.hex_diameter
            n = diameter // 2
            self.action_indices = [(q, r, s) for r in range(-n, n+1) for s in range(n, -(n+1), -1) for q in range(-n, n+1) if q + r + s == 0]
            self.action_to_pixel = lambda action: self._hex_to_pixel(self.action_indices[action])
            self.pixel_to_action = lambda pixel: self.action_indices.index(self._pixel_to_hex(pixel))
            self.get_cell_vertices = lambda position: self._get_hexagon_vertices(position)

            # Compute the total width and height of the board
            if self.orientation == 'flat':
                hex_width = 2 * self.cell_size
                hex_height = math.sqrt(3) * self.cell_size

                half_cells_across = diameter // 2
                full_cells_across = diameter - half_cells_across
                
                self.total_width = (hex_width * full_cells_across) + (hex_width * half_cells_across / 2)
                self.total_height = hex_height * diameter

            elif self.orientation == 'pointy':
                hex_width = math.sqrt(3) * self.cell_size
                hex_height = 2 * self.cell_size

                half_cells_across = diameter // 2
                full_cells_across = diameter - half_cells_across

                self.total_width = hex_width * diameter
                self.total_height = (hex_height * full_cells_across) + (hex_height * half_cells_across / 2)

        elif self.game_info.board_shape == BoardShapes.HEX_RECTANGLE:
            # The canonical 'hex rectangle' is always 'pointy'
            self.orientation = 'pointy'
            height, width = self.game_info.board_dims
            self.action_indices = [(y, x) for y in range(height) for x in range(width)]
            self.action_to_pixel = lambda action: self._hex_rectangle_to_pixel(self.action_indices[action])
            self.pixel_to_action = lambda pixel: self.action_indices.index(self._pixel_to_hex_rectangle(pixel))
            self.get_cell_vertices = lambda position: self._get_hexagon_vertices(position)


            self.hex_width = math.sqrt(3) * self.cell_size
            self.hex_height = 2 * self.cell_size

            # Each row adds a half-cell to the width
            self.total_width = (self.hex_width * width) + ((height - 1) * self.hex_width / 2)

            half_cells_up = height // 2
            full_cells_up = height - half_cells_up
            self.total_height = (self.hex_height * full_cells_up) + (self.hex_height * half_cells_up / 2)

        else:
            raise ValueError(f"Unknown board shape: {self.game_info.board_shape}")
        
        # Add padding to the total width and height
        self.total_width += self.padding
        self.total_height += self.padding
    

    def _grid_to_pixel(self, grid_point):
        '''
        Converts grid coordinates to pixel coordinates
        '''

        # Grid points take the form (y, x)
        x, y = grid_point[1] * self.cell_size, grid_point[0] * self.cell_size

        # Offset by just the padding amount and center in the cell
        x += self.padding / 2 + self.cell_size / 2
        y += self.padding / 2 + self.cell_size / 2
        
        return x, y
    
    def _pixel_to_grid(self, pixel_point):
        '''
        Converts pixel coordinates to grid coordinates
        '''

        # Pixel points take the form (x, y)
        x, y = pixel_point

        # Reverse the padding offset
        x -= self.padding / 2
        y -= self.padding / 2

        # Convert to grid coordinates
        x = x // self.cell_size
        y = y // self.cell_size

        return y, x

    def _hex_to_pixel(self, hex_point):
        '''
        Converts hexagonal coordinates to pixel coordinates

        TODO: had to fix the x, y ordering in grid_to_pixel and pixel_to_grid, so will
              probably need to fix this as well
        '''

        q, r, s = hex_point

        if self.orientation == 'flat':
            x = self.cell_size * (3/2) * q
            y = self.cell_size * (math.sqrt(3)/2 * q + math.sqrt(3) * r)

        elif self.orientation == 'pointy':
            x = self.cell_size * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
            y = self.cell_size * (3/2) * r

        # Offset the coordinates so that the center of the board is at (0, 0)
        x += self.total_width / 2
        y += self.total_height / 2

        return x, y
    
    def _pixel_to_hex(self, pixel_point):
        x, y = pixel_point

        # Reverse the offset applied in _hex_to_pixel
        x -= self.total_width / 2
        y -= self.total_height / 2

        # Apply the inverse of the hex_to_pixel conversion
        q = ((x * math.sqrt(3) / 3) - (y / 3)) / 50
        r = (y * 2 / 3) / 50
        s = -q - r

        # Round the cube coordinates to the nearest hexagon
        q_round, r_round, s_round = round(q), round(r), round(s)
        q_diff, r_diff, s_diff = abs(q_round - q), abs(r_round - r), abs(s_round - s)

        if q_diff > r_diff and q_diff > s_diff:
            q_round = -r_round - s_round

        elif r_diff > s_diff:
            r_round = -q_round - s_round

        else:
            s_round = -q_round - r_round

        return q_round, r_round, s_round
    
    def _hex_rectangle_to_pixel(self, hex_point):
        x, y = hex_point[1] * self.hex_width, hex_point[0] * self.hex_height * 0.75

        x += hex_point[0] * (self.hex_width / 2)

        x += (self.padding / 2) + (self.hex_width / 2)
        y += (self.padding / 2) + (self.hex_height / 2)

        return x, y
    
    def _pixel_to_hex_rectangle(self, pixel_point):
        x, y = pixel_point

        x -= (self.padding / 2) + (self.hex_width / 2)
        y -= (self.padding / 2) + (self.hex_height / 2)

        q = ((x * math.sqrt(3) / 3) - (y / 3)) / self.cell_size
        r = (y * 2 / 3) / self.cell_size
        q_round, r_round = round(q), round(r)

        return r_round, q_round

    def _get_grid_vertices(self, position):
        '''
        Returns the vertices of a square at the given position
        '''
        x, y = position

        points = [
            (x - self.cell_size / 2, y - self.cell_size / 2),
            (x + self.cell_size / 2, y - self.cell_size / 2),
            (x + self.cell_size / 2, y + self.cell_size / 2),
            (x - self.cell_size / 2, y + self.cell_size / 2)
        ]

        return points

    def _get_hexagon_vertices(self, position):
        '''
        Returns the vertices of a hexagon at the given position
        '''
        x, y = position

        if self.orientation == 'flat':
            width = 2 * self.cell_size
            height = math.sqrt(3) * self.cell_size

            points = [
                (x - 0.25 * width, y - 0.5 * height),
                (x + 0.25 * width, y - 0.5 * height),
                (x + 0.5 * width, y),
                (x + 0.25 * width, y + 0.5 * height),
                (x - 0.25 * width, y + 0.5 * height),
                (x - 0.5 * width, y)
            ]

        elif self.orientation == 'pointy':
            width = math.sqrt(3) * self.cell_size
            height = 2 * self.cell_size

            points = [
                (x - 0.5 * width, y - 0.25 * height),
                (x, y - 0.5 * height),
                (x + 0.5 * width, y - 0.25 * height),
                (x + 0.5 * width, y + 0.25 * height),
                (x, y + 0.5 * height),
                (x - 0.5 * width, y + 0.25 * height)
            ]

        return points
    
    def render(self, state, add_button=True, show_legal_actions=True):

        board = state.game_state.board
        legal_actions = state.legal_action_mask

        # Initialize drawing and draw boarder
        drawing = svgwrite.Drawing(filename=RENDER_CONFIG['output_filename'], size=(self.total_width, self.total_height), id="game_board")
        drawing.add(drawing.rect(insert=(0, 0), size=(self.total_width, self.total_height), stroke='black', stroke_width=1, fill='none'))

        for i, occupant in enumerate(board):
            position = self.action_to_pixel(i)
            vertices = self.get_cell_vertices(position)

            # Draw the cell
            drawing.add(drawing.polygon(vertices, fill=self.render_config["light_blue"], stroke=self.render_config["light_grey"], stroke_width=1))

            # Draw the legal action mask
            if legal_actions[i] and show_legal_actions:
                drawing.add(drawing.circle(center=position, r=self.render_config['legal_radius'], fill=self.render_config['purple'], stroke=self.render_config['dark_grey'], stroke_width=1))

            # Draw the piece (if present)
            if occupant == P1:
                drawing.add(drawing.circle(center=position, r=self.render_config['piece_radius'], fill=self.render_config['off_white'], stroke=self.render_config['dark_grey'], stroke_width=1))
            elif occupant == P2:
                drawing.add(drawing.circle(center=position, r=self.render_config['piece_radius'], fill=self.render_config['off_black'], stroke=self.render_config['dark_grey'], stroke_width=1))

        # Add an invisible rectangle to capture the user's clicks
        if add_button:
            drawing.add(drawing.rect(insert=(0, 0), size=(self.total_width, self.total_height), class_="btn", onclick="handleClick(event)"))

        drawing.save()