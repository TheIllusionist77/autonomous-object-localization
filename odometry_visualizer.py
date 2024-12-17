import pygame, os, math, random, time

pygame.init()

WIDTH = 1280
HEIGHT = 720
SCALE = 8
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
    objects_locations = {}
    last_disp = 0

    for line in log_data:
        parts = line.strip().split(", ")
        
        disp = -float(parts[0][1:]) + float(parts[1]) + float(parts[2]) - last_disp
        last_disp = disp + last_disp

        yaw = float(parts[3])
        tof_reading = float(parts[4])
        if parts[5][:-1] != "None":
            object = parts[5][:-1]
        else:
            object = None

        path.append((disp, tof_reading, yaw, object))
        
        if object != None and object not in objects_locations:
            objects_locations[object] = []
            objects_locations[object].append([disp, yaw, tof_reading])
        elif object in objects_locations:
            objects_locations[object].append([disp, yaw, tof_reading])

    objects = {}
    for object in objects_locations:
        object_x, object_y = 0, 0
        count = 0

        for each in objects_locations[object]:
            count += 1
            disp, tof, yaw = each

            yaw *= -1

            if count == 1:
                initial_angle = yaw - 90

            yaw -= initial_angle

            object_x += (disp + int(-tof / 3.4 * math.cos(yaw))) * SCALE
            object_y += (disp + int(-tof / 3.4 * math.sin(yaw))) * SCALE

        object_x /= count
        object_y /= count
        objects[object] = object_x, object_y
    
    return path, objects

def draw_path(path, objects):
    SCREEN.fill((255, 255, 255))
    x, y = WIDTH / 2, HEIGHT / 2
    count = 0

    for point in path:
        count += 1
        disp, tof, yaw, obj = point
        yaw *= -1

        if count == 1:
            initial_angle = yaw - 90

        yaw -= initial_angle
        
        x += float(disp * math.cos(math.radians(yaw))) * SCALE
        y += float(disp * math.sin(math.radians(yaw))) * SCALE

        pygame.draw.circle(SCREEN, (0, min(255, count), 255), (x, y), 1)
        pygame.display.update()

    for object in objects:
        object_x, object_y = objects[object]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        pygame.draw.circle(SCREEN, color, (WIDTH / 2 + object_x, HEIGHT / 2 + object_y), 5)

        font = pygame.font.SysFont("Arial", 12)
        text = font.render(object, True, color)
        SCREEN.blit(text, (WIDTH / 2 + object_x + 10, HEIGHT / 2 + object_y - 15))
        pygame.display.update()

log_data = read_file(log_directory)

if log_data is None:
    print("No log files found.")
    running = False
else:
    path, objects = parse_data(log_data)
    running = True

drawn = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not drawn:
        draw_path(path, objects)
        drawn = True

pygame.quit()
