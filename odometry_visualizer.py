import pygame, os, random, ast

pygame.init()

WIDTH = 1280
HEIGHT = 720
SCALE = 10
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

log_directory = "/Users/sanjitprakash/Downloads"

def read_file(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as file:
                    return file.readlines()
            except Exception as e:
                print("Error reading file.")
    return None

def parse_data(log_data):
    path = []

    for line in log_data:
        parts = line.strip().split(", ")
        x = parts[0][1:]
        y = parts[1]

        if line[:1] == "[":
            break

        path.append((x, y, parts[2][:-1]))

    object_locations = ast.literal_eval(line)
    print(object_locations)
    return path, object_locations

def draw_path(path, object_locations):
    SCREEN.fill((0, 0, 0))
    center_x, center_y = WIDTH / 2, HEIGHT / 2
    count = 0

    for point in path:
        count += 1
        disp_x, disp_y, name = point

        x = (float(disp_x) * SCALE) + center_x
        y = (float(disp_y) * SCALE) + center_y

        print(x, y)

        pygame.draw.circle(SCREEN, (0, min(255, count), 255), (x, y), 1)
        pygame.display.update()

    for object in object_locations:
        name, x, y, last_seen = object

        x = (float(disp_x) * SCALE) + center_x
        y = (float(disp_y) * SCALE) + center_y

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        pygame.draw.circle(SCREEN, color, (x, y), 5)

        font = pygame.font.SysFont("Arial", 12)
        text = font.render(name, True, color)
        SCREEN.blit(text, (x + 10, y - 15))
        pygame.display.update()

log_data = read_file(log_directory)

if log_data is None:
    print("No log files found.")
    running = False
else:
    path, object_locations = parse_data(log_data)
    running = True

drawn = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not drawn:
        draw_path(path, object_locations)
        drawn = True

pygame.quit()
