import pygame, os, math, random

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
                print(f"Error reading file.")
    return None

def parse_data(log_data):
    path = []
    objects_locations = {}  # Dictionary to store locations for objects

    for line in log_data:
        parts = line.strip().split(", ")
        
        y_disp = float(parts[1])
        yaw = float(parts[3])
        tof_reading = float(parts[4])
        if parts[5][:-1] != "None":
            object = parts[5][:-1]
        else:
            object = None

        path.append((y_disp, yaw, tof_reading, object))
        
        if object != None and object not in objects_locations:
            objects_locations[object] = []
            objects_locations[object].append((y_disp, yaw, tof_reading))

    averaged_objects_locations = {}
    for obj, locations in objects_locations.items():
        avg_x, avg_y = 0, 0
        for loc in locations:
            y_disp, yaw, tof = loc
            avg_x += int(tof * math.cos(yaw)) / SCALE
            avg_y += int(tof * math.sin(yaw)) / SCALE

        averaged_objects_locations[obj] = (avg_x / len(locations), avg_y / len(locations))
    
    return path, averaged_objects_locations

def draw_path(path, averaged_objects_locations):
    SCREEN.fill((0, 0, 0))

    x, y = WIDTH / 4, HEIGHT / 4

    for point in path:
        y_disp, tof, yaw, obj = point
        print(x, y)
        x += int(y_disp * math.cos(yaw)) / SCALE
        y += int(y_disp * math.sin(yaw)) / SCALE

        pygame.draw.circle(SCREEN, (0, 0, 255), (x, y), 1)

    for obj, (avg_x, avg_y) in averaged_objects_locations.items():
        object_x = x + int(avg_x * math.cos(yaw))
        object_y = y + int(avg_y * math.sin(yaw))

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        pygame.draw.circle(SCREEN, color, (object_x, object_y), 5)

        font = pygame.font.SysFont("Arial", 12)
        text = font.render(obj, True, color)
        SCREEN.blit(text, (object_x + 10, object_y - 15))

log_data = read_file(log_directory)

if log_data is None:
    print("No log files found.")
    running = False
else:
    path, averaged_objects_locations = parse_data(log_data)
    running = True

draw_path(path, averaged_objects_locations)
pygame.display.update()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()