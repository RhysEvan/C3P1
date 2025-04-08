import wx
import os
import glob
from PIL import Image
import numpy as np
import time
import cv2
from Classes.Multicam import *



class ImageDisplayFrame(wx.Frame):
    def __init__(self, parent, title, image_files, output_folder, camera,delay):
        self.image_files = image_files
        self.output_folder = output_folder
        self.camera = camera
        self.current_image_index = -1
        self.current_bmp = wx.NullBitmap
        self.delay = delay

        num_displays = wx.Display.GetCount()
        display_index = 1 if num_displays >= 2 else 0
        if display_index == 0:
            print("Warning: Only one display detected. Displaying on primary.")
        else:
            print("Multiple displays detected. Using second display.")
        display = wx.Display(display_index)
        geometry = display.GetGeometry()

        super().__init__(parent, title=title, style=wx.NO_BORDER)
        self.SetPosition(geometry.GetTopLeft())
        self.ShowFullScreen(True, style=wx.FULLSCREEN_ALL)

        self.panel = wx.Panel(self, style=wx.FULL_REPAINT_ON_RESIZE)
        self.panel.SetBackgroundColour(wx.BLACK)
        self.panel.SetSize(self.GetClientSize())  # Set the size of the panel to the size of the screen
        self.panel.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        # Ensure the panel occupies the full frame
        self.Bind(wx.EVT_SIZE, self.OnSize)

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.panel.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)
        self.timer.Start(0)  # Start the timer with a 500ms interval

    def OnSize(self, event):
        # Resize the panel to occupy the full frame
        self.panel.SetSize(self.GetClientSize())
        event.Skip()

    def OnTimer(self, event):
        self.ShowNextImage()

    def ShowNextImage(self):
        self.current_image_index += 1

        if self.current_image_index>= len(self.image_files):
            print("Finished displaying all images.")
            self.Close()
            return

        filepath = self.image_files[self.current_image_index]
        print(f"Displaying: {os.path.basename(filepath)}")

        try:
            img = wx.Image(filepath, wx.BITMAP_TYPE_ANY)
            if not img.IsOk():
                print(f"Error: Failed to load image {filepath}")
                self.current_bmp = wx.NullBitmap
                wx.CallAfter(self.ShowNextImage)
                return

            screen_w, screen_h = self.GetClientSize()
            img_w, img_h = img.GetWidth(), img.GetHeight()

            if img_w <= 0 or img_h <= 0:
                print(f"Error: Invalid image dimensions for {filepath}")
                self.current_bmp = wx.NullBitmap
                wx.CallAfter(self.ShowNextImage)
                return

            # Calculate the scaling factor to fit the image within the screen while maintaining aspect ratio
            scale_w = screen_w / img_w
            scale_h = screen_h / img_h
            scale = min(scale_w, scale_h)

            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            if new_w <= 0 or new_h <= 0:
                new_w, new_h = 1, 1

            img = img.Scale(new_w, new_h, wx.IMAGE_QUALITY_HIGH)
            self.current_bmp = wx.Bitmap(img)

            print("Refreshing panel for repaint...")
            self.panel.Refresh()
            self.panel.Update()

            # if framenumber is not zero
            if self.camera:
                #wx.CallLater(500,self.CaptureAndSaveFrame,filepath)
                time.sleep(self.delay/1000)

                self.CaptureAndSaveFrame(filepath)
            else:
                print("Skipping camera capture as no camera object was provided or it failed.")

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            self.current_bmp = wx.NullBitmap
            self.panel.Refresh()
            wx.CallAfter(self.ShowNextImage)

    def OnPaint(self, event):
        if not self.panel:
            print("Error: Panel is not initialized.")
            return

        try:
            dc = wx.BufferedPaintDC(self.panel)
        except:
            print("Error: Failed to draw buffer.")
            dc = None
        if not dc:
            print("Error: Failed to create BufferedPaintDC")
            return

        dc.SetBackground(wx.Brush(self.panel.GetBackgroundColour()))
        dc.Clear()

        if self.current_bmp and self.current_bmp.IsOk():
            panel_w, panel_h = self.panel.GetClientSize()
            bmp_w, bmp_h = self.current_bmp.GetSize()

            pos_x = max(0, (panel_w - bmp_w) // 2)
            pos_y = max(0, (panel_h - bmp_h) // 2)
            dc.DrawBitmap(self.current_bmp, pos_x, pos_y, useMask=False)

    def CaptureAndSaveFrame(self, displayed_image_path):
        if not self.camera:
            print("CaptureAndSaveFrame called, but camera object is None.")
            return
        try:
            if isinstance(self.camera, MultiCam):
                cam_frame = self.camera.GetFrame()
            else:
                frame_data = self.camera.GetFrame()
                if isinstance(frame_data, np.ndarray):
                    cam_frame = Image.fromarray(frame_data)
                elif isinstance(frame_data, Image.Image):
                    cam_frame = frame_data
                else:
                    print(f"Error: Cam.GetFrame() returned unexpected type: {type(frame_data)}")
                    return

            if not cam_frame:
                print("Error: Received no valid frame from camera.")
                return

            os.makedirs(self.output_folder, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(displayed_image_path))[0]
            output_filename = os.path.join(self.output_folder, f"{base_name}_capture.png")

            idx = 0
            for frame in cam_frame:
                fold = os.path.join(self.output_folder, str(idx))
                os.makedirs(fold, exist_ok=True)
                output_filename = os.path.join(fold ,f"{base_name}_capture.png")
                frame.save(output_filename)
                print(f"Saving camera frame to: {output_filename}")

                idx += 1
            #plot frame with opencv
            # Convert PIL images to OpenCV format
            opencv_image1 = cv2.cvtColor(np.array(cam_frame[0]), cv2.COLOR_RGB2BGR)
            opencv_image2 = cv2.cvtColor(np.array(cam_frame[1]), cv2.COLOR_RGB2BGR)

            # Rescale images to a height of 640 while maintaining aspect ratio
            height = 640
            scale1 = height / opencv_image1.shape[0]
            scale2 = height / opencv_image2.shape[0]
            width1 = int(opencv_image1.shape[1] * scale1)
            width2 = int(opencv_image2.shape[1] * scale2)

            resized_image1 = cv2.resize(opencv_image1, (width1, height))
            resized_image2 = cv2.resize(opencv_image2, (width2, height))

            # Concatenate images side by side
            concatenated_image = np.hstack((resized_image1, resized_image2))

            # Display the concatenated image
            cv2.imshow("Concatenated Frame", concatenated_image)
            cv2.waitKey(0)
            #continue


        except AttributeError as e:
            print(f"Error: Camera object missing 'GetFrame' method or issue during capture: {e}")
        except ImportError as e:
            print(f"Import Error during capture/save (maybe missing Pillow or OpenCV?): {e}")
        except Exception as e:
            print(f"Error capturing or saving camera frame: {e}")

    def OnClose(self, event):
        print("Closing application window...")
        cv2.destroyAllWindows()
        self.timer.Stop()
        self.Destroy()

class MyApp(wx.App):
    def __init__(self, input_folder, output_folder, allowed_extensions, camera_obj,delay):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.allowed_extensions = allowed_extensions
        self.camera = camera_obj
        self.delay = delay
        super().__init__(False)

    def OnInit(self):
        print("MyApp OnInit called.")
        if not os.path.isdir(self.input_folder):
            print(f"Error: Input folder '{self.input_folder}' not found.")
            wx.MessageBox(f"Input folder not found:\n{self.input_folder}", "Error", wx.OK | wx.ICON_ERROR)
            return False
        try:
            os.makedirs(self.output_folder, exist_ok=True)
        except OSError as e:
            print(f"Error creating output folder '{self.output_folder}': {e}")
            wx.MessageBox(f"Could not create output folder:\n{self.output_folder}\n{e}", "Error", wx.OK | wx.ICON_ERROR)
            return False

        image_files = []
        for ext in self.allowed_extensions:
            pattern = os.path.join(self.input_folder, ext)
            image_files.extend(glob.glob(pattern))

        if not image_files:
            ext_str = ", ".join(self.allowed_extensions)
            print(f"Error: No images found in '{self.input_folder}' with extensions {ext_str}")
            wx.MessageBox(f"No images found in:\n{self.input_folder}\n\nAllowed extensions:\n{ext_str}", "Error", wx.OK | wx.ICON_INFORMATION)
            return False

        print(f"Found {len(image_files)} images.")
        image_files.sort()

        self.frame = ImageDisplayFrame(None, "Image Viewer", image_files, self.output_folder, self.camera,self.delay)
        self.frame.Show()
        return True

    def OnExit(self):
        print("MyApp OnExit called.")
        cv2.destroyAllWindows()


        return 0

if __name__ == "__main__":
    INPUT_FOLDER = r"C:\Users\Seppe\PycharmProjects\phase\phase-shifting\output_num"
    INPUT_FOLDER = r"C:\Users\Seppe\PycharmProjects\phase\phase-shifting\sample_data\gamma_correction"
    INPUT_FOLDER = r"C:\Users\Seppe\PycharmProjects\phase\phase-shifting\output"

    OUTPUT_FOLDER = r"C:\Users\Seppe\PycharmProjects\phase\phase-shifting\sample_data\objectSgamma"
    OUTPUT_FOLDER = r"C:\Users\Seppe\PycharmProjects\phase\phase-shifting\sample_data\objectS3"
    ALLOWED_EXTENSIONS = ('*.png',)

    camera_instance = None
    try:
        camera_instance = MultiCam()
        camera_instance.load_cams_from_file('camera_config.json')

        print("Camera initialized successfully.")
    except Exception as e:
        print(f"Error initializing camera: {e}. Proceeding without camera capture.")

    app = MyApp(input_folder=INPUT_FOLDER,
                output_folder=OUTPUT_FOLDER,
                allowed_extensions=ALLOWED_EXTENSIONS,
                camera_obj=camera_instance,
                delay=70)
    print("Starting wxPython MainLoop...")
    app.MainLoop()
    print("Application finished.")
