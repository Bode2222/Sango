import os
import pygame
from GenFromImg import img_to_tuple
from WaveFuncCollapse import Rule, WFCollapse2D, Dir, SimpleWeightGen, BasicWeightGen, extractRulesAndRelativeFrequencies2D


# Define some colors
BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)
GREEN = ( 0, 255, 0)
RED = ( 255, 0, 0)
BLUE = (0, 0, 255)
BROWN = (165, 42, 42)
SAND = (194, 178, 128)
base_colors = [(0, 0, 0), (50, 50, 50)]

# Takes in the grid and a list of colors the same length as the number of unique tiles in the grid. prepend the color for an unsolved tile into the tiled_colors list
def draw2D(grid, tile_colors, screen):
	# Define constants
	CELL_WID = int(.8 * screen.get_height() / grid.wid)
	CUSHION = int(.05 * screen.get_width() / grid.wid)

	# Draw rects that represent the grid
	for y in range(grid.wid):
		for x in range(grid.len):
			#rect = pygame.Rect(x * (CELL_WID + CUSHION), y * (CELL_WID + CUSHION), CELL_WID, CELL_WID, tile_colors[grid.get_pos([x, y]).chosen_tile + 1])
			color = tile_colors[grid.get_cell([x, y]).chosen_tile + 2]
			pygame.draw.rect(screen, color, [x * (CELL_WID + CUSHION), y * (CELL_WID + CUSHION), CELL_WID, CELL_WID], 0)


if __name__ == '__main__':
	# init game engine
	pygame.init()

	# Set up algo hyperparams
	dims = [35, 35]
	con_dims = [3, 3]

	# What image to use as the base image
	cur_path = os.path.dirname(__file__)
	path = os.path.join(cur_path, "samples\\Flowers.png")
	# Change image to tuple
	img_grid, col_list = img_to_tuple(str(path))
	# extract rules, frequencies and mapping from img tuple
	rules, frequencies, _ = extractRulesAndRelativeFrequencies2D(img_grid)
	# Set up wf collapse algo around those rules and frequencies
	test = WFCollapse2D(dims=dims, tile_selection="LINEAR",n_tiles=int(len(frequencies)/4), rules=rules, 
		weight_genner=SimpleWeightGen(frequencies, con_dims), context_dims=con_dims)

	# Used to know whether original image should be displayed to screen
	toggle_orig = False

	# open a new window
	screen = pygame.display.set_mode((750, 500))
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
			draw2D(img_grid, base_colors + col_list, screen)
		else:
			draw2D(test._grid, base_colors + col_list, screen)
	
	
		# --- Go ahead and update the screen with what we've drawn.
		pygame.display.flip()
		
		# --- Limit to 60 frames per second
		clock.tick(1600000)
 
	#Once we have exited the main program loop we can stop the game engine:
	pygame.quit()