import pygame
import cv2
import numpy as np
from threading import Thread, Event
import time

def create_cross_image(width, height):
    """
    Create an image with a white background and a black cross in the middle.
    """
    print(width)
    print(height)
    image = np.ones((width, height, 3), dtype=np.uint8) * 255  # White background
    thickness = 10  # Thickness of the cross lines

    # Draw horizontal line
    cv2.line(image, (height // 2, 0), (height // 2, width), (0, 0, 0), thickness)
    # Draw vertical line
    cv2.line(image, (0, width // 2), (height, width // 2), (0, 0, 0), thickness)

    return image

def show_camera_feed_with_marker(cameras, dimensions, stop_event):
    while not stop_event.is_set():
        for i, camera in enumerate(cameras):
            frame = camera.GetFrame()
            frame = frame.astype(np.uint8)

            # Convert to BGR for display
            if len(frame.shape) == 2:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 3:
                bgr_frame = frame
            else:
                raise ValueError("Unsupported frame format")

            # Draw marker in the center of the frame
            center_x, center_y = bgr_frame.shape[1] // 2, bgr_frame.shape[0] // 2
            marker_size = 10
            cv2.drawMarker(bgr_frame, (center_x, center_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=marker_size, thickness=2)

            # Resize to fit the specified dimensions
            bgr_frame = cv2.resize(bgr_frame, dimensions[i], interpolation=cv2.INTER_AREA)

            # Show in OpenCV window
            window_name = f'Camera Feed {i+1}' if len(cameras) > 1 else 'Camera Feed'
            cv2.imshow(window_name, bgr_frame)

        if cv2.waitKey(1) & 0xFF == 13:  # Break loop on Enter key
            stop_event.set()
            break

    cv2.destroyAllWindows()

def pygame_event_loop(stop_event, screen, cross_image):
    """Handles Pygame events and displays the cross image."""
    pygame.event.get()

    # Display the cross image
    cross_surface = pygame.surfarray.make_surface(cross_image)
    screen.blit(cross_surface, (0, 0))
    pygame.display.flip()

    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                break
        time.sleep(0.01)  # Small delay to reduce CPU usage

def innitiate_support(cameras, dimensions):
    pygame.init()
    num_displays = pygame.display.get_num_displays()
    desktop_sizes = pygame.display.get_desktop_sizes()
    if num_displays < 2:
        screen_width, screen_height = desktop_sizes[0]
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN, display=0)
    else:
        screen_width, screen_height = desktop_sizes[1]
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN, display=1)

    cross_image = create_cross_image(screen_width, screen_height)

    # Create a stop event
    stop_event = Event()

    # Start Pygame event loop in a separate thread
    thread_pygame = Thread(target=pygame_event_loop, args=(stop_event, screen, cross_image))
    thread_pygame.start()

    # Start camera feed with marker in a separate thread
    thread_camera = Thread(target=show_camera_feed_with_marker, args=(cameras, dimensions, stop_event))
    thread_camera.start()

    # Wait for both threads to complete
    thread_pygame.join()
    thread_camera.join()

    pygame.quit()
