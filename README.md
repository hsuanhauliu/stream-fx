# StreamFX

Custom and real-time video effects for your webcam feed.

Webcam &#8594; **StreamFX** &#8594; OBS Virtual Camera &#8594; FaceTime

This program is designed to work with OBS on MacOS, but it should work on Windows and Linux. The program starts a MJPEG stream and serves post-processed frames of your camera feed. This design allows more flexibility and avoids using the popular pyvirtualcam library, since it does not work on newer version of MacOS due to this [bug](https://github.com/letmaik/pyvirtualcam/issues/111).

## Usage

Install and run StreamFX.

```bash
poetry install
poetry run stream-fx

# see input args
poetry run stream-fx --help
```

Configure your OBS studio.

1. Add the server URL as a new source.
    1. Select Media Source: From the menu that appears, choose "Media Source".
    1. You can give it a descriptive name like "Python Stream" and click "OK".
    1. Configure the Properties: A new window will open for the source's properties. This is the most important step:
        1. Uncheck the box that says "Local File".
        1. In the "Input" field, enter the URL from your script: <http://127.0.0.1:8080/video_feed>.
        1. Click "OK".
    1. You should see the video feed on the OBS video panel. If not, try restarting StreamFX program.
1. Click "Start Virtual Camera" to start sending video feed from your source to the virtual cam. If this is your first time using OBS virtual camera (on MacOS), you'll be asked to change your privacy settings to allow OBS to install virtual cam on your system. Simply follow instructions in the pop-up window to complete the installation.

Once your virtual camera is started, the processed video frames should be visible on your virtual camera. You can test this by using any video call program and set your camera input to the virtual cam. For FaceTime, click on "Video" dropdown in the top menu, and select "OBS Virtual Camera".

To stop the video feed, simply stop the StreamFX program.

### Settings

The program takes in a YAML format config file to easily configure the app. See [example](./example_config.yaml).

### Control Panel

StreamFX includes a lightweight web UI control panel.

Default URL: <http://127.0.0.1:8080/>
