import pygame
from WaveFuncCollapse import Rule, WFCollapse2D, Dir, SimpleWeightGen, BasicWeightGen, extractRulesAndRelativeFrequencies2D



# Define some colors
BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)
GREEN = ( 0, 255, 0)
RED = ( 255, 0, 0)
BLUE = (0, 0, 255)
BROWN = (165, 42, 42)
SAND = (194, 178, 128)

# Takes in the grid and a list of colors the same length as the number of unique tiles in the grid. prepend the color for an unsolved tile into the tiled_colors list
def draw2D(grid, tile_colors, screen):
	# Define constants
	CELL_WID = int(.95 * screen.get_height() / grid.wid)
	CUSHION = int(.05 * screen.get_width() / grid.wid)

	# Draw rects that represent the grid
	for y in range(grid.wid):
		for x in range(grid.len):
			#rect = pygame.Rect(x * (CELL_WID + CUSHION), y * (CELL_WID + CUSHION), CELL_WID, CELL_WID, tile_colors[grid.get_pos([x, y]).chosen_tile + 1])
			color = tile_colors[grid.get_cell([x, y]).chosen_tile + 2]
			pygame.draw.rect(screen, color, [x * (CELL_WID + CUSHION), y * (CELL_WID + CUSHION), CELL_WID, CELL_WID], 0)

if __name__ == '__main__':
	# Set random seed for consistent results
	#random.seed(a=1235)
	# init game engine
	pygame.init()


	# Set up algo
	dims = [35, 35]
	con_dims = [3, 3]
	# Set generation rules. 0, 1, 2 = Land, sea, coast
	rules = [
		Rule(0, 2, Dir.UP), Rule(0, 2, Dir.DOWN), Rule(0, 2, Dir.LEFT), Rule(0, 2, Dir.RIGHT),
		Rule(0, 0, Dir.UP), Rule(0, 0, Dir.DOWN), Rule(0, 0, Dir.LEFT), Rule(0, 0, Dir.RIGHT),
		Rule(1, 2, Dir.UP), Rule(1, 2, Dir.DOWN), Rule(1, 2, Dir.LEFT), Rule(1, 2, Dir.RIGHT),
		Rule(1, 1, Dir.UP), Rule(1, 1, Dir.DOWN), Rule(1, 1, Dir.LEFT), Rule(1, 1, Dir.RIGHT),
		Rule(2, 0, Dir.UP), Rule(2, 0, Dir.DOWN), Rule(2, 0, Dir.LEFT), Rule(2, 0, Dir.RIGHT),
		Rule(2, 1, Dir.UP), Rule(2, 1, Dir.DOWN), Rule(2, 1, Dir.LEFT), Rule(2, 1, Dir.RIGHT),
		Rule(2, 2, Dir.UP), Rule(2, 2, Dir.DOWN), Rule(2, 2, Dir.LEFT), Rule(2, 2, Dir.RIGHT)
	]
	# Create Wave-funciton collapse object
	wf = WFCollapse2D(dims, 3, rules, weight_genner=BasicWeightGen([5, 3, 5]), context_dims=con_dims)
	while wf.step(): pass
	# CREATE COPY
	rules, frequencies, mapping = extractRulesAndRelativeFrequencies2D(wf._grid)
	test = WFCollapse2D(dims=dims, n_tiles=int(len(frequencies)/4), rules=rules, weight_genner=SimpleWeightGen(frequencies, con_dims), context_dims=con_dims)
	toggle_orig = False

	# transfer colors from original to copy
	color_map = {0: GREEN, 1: BLUE, 2: SAND}
	orig_colors = [(0, 0, 0), (100, 100, 100), color_map[0], color_map[1], color_map[2]]
	test_colors = [(0, 0, 0) for x in range(int(len(frequencies)/4) + 2)]
	test_colors[1] = (100, 100, 100)
	keys = list(mapping.keys())
	vals = list(mapping.values())
	for value in vals:
		i = vals.index(value)
		orig_tile = keys[i]
		if (orig_tile >= 0):
			test_colors[orig_tile+2] = color_map[orig_tile]

	# open a new window
	screen = pygame.display.set_mode((500, 500))
	pygame.display.set_caption("Wave Function Collapse")

	# Game loop run while 'running' is true
	running = True
	# to limit framerate
	clock = pygame.time.Clock()

	# -------------------- Main Game Loop --------------------
	while running:
		# --- Main event loop
		for event in pygame.event.get(): # User did something
			if event.type == pygame.QUIT: # If user clicked close
				running = False # Flag that we are done so we can exit the while loop
			elif event.type == pygame.MOUSEBUTTONDOWN:
				toggle_orig = not toggle_orig
	
		# --- Game logic should go here
		if not test.step():
			pygame.time.delay(1000)
			test.reset()
	
		# --- Drawing code should go here
		# First, clear the screen to white. 
		screen.fill(WHITE)
		# Then you can draw different shapes and lines or add text to your background stage.
		if toggle_orig:
			draw2D(wf._grid, orig_colors, screen)
		else:
			draw2D(test._grid, test_colors, screen)
	
	
		# --- Go ahead and update the screen with what we've drawn.
		pygame.display.flip()
		
		# --- Limit to 60 frames per second
		clock.tick(1600000)
 
	#Once we have exited the main program loop we can stop the game engine:
	pygame.quit()