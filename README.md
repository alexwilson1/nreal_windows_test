# Nreal Air AR Multiple Screens (Windows 11)
Simple project to get a proof of concept for Nreal Air AR glasses working in Windows. The main purpose of this is to unlock some development in the space.

This was created during some spare personal time, and is in no way endorsed by or associated with any organization.

Use this entirely at your own risk and read the license.

This is not being actively maintained.

## How it works

1. Face detection and angle detection

2. Determining which 'screens' (virtual) the user is 'looking at' and by how much

3. Capturing relevant portions of those 'screens' and combining them into a 1920x1080 image. Adding black borders to parts of the image that are not 'in view'

4. Outputting the image to the Nreal Air display

### Installing

1. Git clone the repo and use conda/mamba to create the environment from the `environment.yml` file

2. Create three virtual displays using this guide:
https://www.amyuni.com/forum/viewtopic.php?t=3030

3. Calibrate variables at the top by printing the `angles` variable and moving your head to up/down/left/right extremes. Take measurements and replace the values at the top of `main.py`

4. Read the `sct.monitors` variable to figure out which monitors are which and update these variable numbers


### Current (known) issues:

1. Framerate/lag/high resource usage - mainly due to ML models being used for head pose estimation. Probably the best method here would be to try and use the gyro from the device itself.
2. Mouse not shown on display - known issue with a few possible solutions https://stackoverflow.com/questions/72328718/python-take-screenshot-including-mouse-cursor
3. Extra screen that is used to display the video output should somehow be disabled from interaction with other windows
4. Hardcoded to work with three virtual screens - could be generalised to 'n' screens
5. Viewport algorithm is basic and does not account for translation or roll
6. Calibration can be an issue - the best way to solve this would be to use the gyro in the device




## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors
Alex Wilson

## Acknowledgments
The head pose estimation algorithm and models come from Yin Guobing (one of the models was modified by conversion to ONNX)
:
https://github.com/yinguobing/head-pose-estimation

The virtual display driver (not included) comes from Amyuni:
https://www.amyuni.com/forum/viewtopic.php?t=3030
