#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.1post4),
    on November 23, 2024, at 00:16
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.1post4'
expName = 'solo project'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'sex': '',
    'age': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1536, 864]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Tanishka\\OneDrive\\Documents\\2. Courses\\PSY310\\Affective Recognition Task\\Affective Recognition Test_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0.0000, 0.0000, 0.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0.0000, 0.0000, 0.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    if deviceManager.getDevice('key_resp_4') is None:
        # initialise key_resp_4
        key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_4',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "consent" ---
    consentform = visual.TextStim(win=win, name='consentform',
        text='This is a study on memory. You are free to exit the experiment at any time if you no longer wish to participate in the experiment. Your personal details will be kept confidential and not be disclosed in the research. If you would like to receive your performance data post the experiment, please email the researcher at tanishka.s@ahduni.edu.in. By pressing the below button, you consent to participating in the experiment.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    button = visual.ButtonStim(win, 
        text='I consent', font='Arvo',
        pos=(0, -0.2),
        letterHeight=0.03,
        size=(0.3, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button',
        depth=-1
    )
    button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "dsr_personal" ---
    text = visual.TextStim(win=win, name='text',
        text='Please indicate how much you agree to the following statement or how true it is about yourself.',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    question = visual.TextStim(win=win, name='question',
        text='',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider = visual.Slider(win=win, name='slider',
        startValue=None, size=(1.15, 0.1), pos=(0, -0.1), units=win.units,
        labels=["Strongly disagree", "Mildly disagree", "Neither disagree nor agree", "Mildly agree", "Strongly agree"], ticks=(0, 1, 2, 3, 4), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.025,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    
    # --- Initialize components for Routine "dsr_experience" ---
    text2 = visual.TextStim(win=win, name='text2',
        text='How disgusting would you find the following experience?',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    question2 = visual.TextStim(win=win, name='question2',
        text='',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider_2 = visual.Slider(win=win, name='slider_2',
        startValue=None, size=(1.15, 0.1), pos=(0, -0.1), units=win.units,
        labels=["Not disgusting at all", "Slightly disgusting", "Moderately disgusting", "Very disgusting", "Extremely disgusting"], ticks=(0, 1, 2, 3, 4), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.025,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    
    # --- Initialize components for Routine "instr" ---
    instruction = visual.TextStim(win=win, name='instruction',
        text='Congratulations. You have successfully finished the survey. Now, its time for the experiment. \n\nPay attention to both the image and the word presented on the screen. First, look at the image and then the word.\n\nPress SPACE to begin experiment.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "encoding" ---
    fixation = visual.ShapeStim(
        win=win, name='fixation', vertices='cross',
        size=(0.02, 0.02),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    study_image = visual.ImageStim(
        win=win,
        name='study_image', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.50, 0), draggable=False, size=(0.75, 0.75),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    studyword = visual.TextStim(win=win, name='studyword',
        text='',
        font='Open Sans',
        units='norm', pos=(0.50, 0), draggable=False, height=0.12, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "test_instruction" ---
    test_instr = visual.TextStim(win=win, name='test_instr',
        text="In this task, you will be presented with words. If the word was presented in this study, press 'Old.' If the word was not presented in this study, press 'New.'\n\nPress SPACE to begin experiment.",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "testing" ---
    t_word = visual.TextStim(win=win, name='t_word',
        text='',
        font='Open Sans',
        pos=(0, 0.3), draggable=False, height=0.12, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    slider_3 = visual.Slider(win=win, name='slider_3',
        startValue=None, size=(1.0, 0.1), pos=(0, -0.2), units=win.units,
        labels=["Old", "New"], ticks=(1, 2), granularity=1.0,
        style='slider', styleTweaks=(), opacity=None,
        labelColor='White', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Lucida Sans Unicode Regular', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    
    # --- Initialize components for Routine "end" ---
    thanks = visual.TextStim(win=win, name='thanks',
        text="Are you an online participant? \n\nIf yes, press 'Yes.' \nIf no, press the 'ESC' button on the keyboard.",
        font='Open Sans',
        pos=(0, 0.05), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    button_2 = visual.ButtonStim(win, 
        text='Yes', font='Arvo',
        pos=(0, -0.1),
        letterHeight=0.03,
        size=(0.3, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_2',
        depth=-1
    )
    button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "brk" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text='We are almost at the end of the experiment. There is one last task remaining. If you would like, you can take a short break. Just ensure that you do not close the screen or press the escape button. \n\nWhen you are ready for the last task, press SPACE.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    
    # --- Initialize components for Routine "lsas_instr" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text='This is a short survey. Reach each situation carefully. The first question asks how anxious or fearful you feel in the situation. The second questions asks how often you avoid the situation. If you come across a situation that you ordinarily do not experience, imagine "what if you were faced with that situation," and then, rate the degree to which you would fear or avoid this hypothetical situation.\n\nPlease base your ratings on the way that the situations have affected you in the last week.\n\nPress SPACE to proceed.\n\nPlease answer the first question first, and then the second. Answering the second question first would move on to the second question without recording the first one. It is important you answer all of them.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    
    # --- Initialize components for Routine "lsas" ---
    question_3 = visual.TextStim(win=win, name='question_3',
        text='',
        font='Open Sans',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_2 = visual.TextStim(win=win, name='text_2',
        text='How anxious or fearful would you feel in this situation?',
        font='Open Sans',
        pos=(0, 0.2), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider_4 = visual.Slider(win=win, name='slider_4',
        startValue=None, size=(1.15, 0.1), pos=(0, 0.1), units=win.units,
        labels=["None", "Mild", "Moderate", "Severe"], ticks=(0, 1, 2, 3), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.025,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    text_3 = visual.TextStim(win=win, name='text_3',
        text='How often do you avoid this situation?',
        font='Open Sans',
        pos=(0, -0.1), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    slider_5 = visual.Slider(win=win, name='slider_5',
        startValue=None, size=(1.15, 0.1), pos=(0, -0.2), units=win.units,
        labels=["Never (0%)", "Occasionally (1-33%)", "Often (34-66%)", "Usually (67-100%)"], ticks=(0, 1, 2, 3), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.025,
        flip=False, ori=0.0, depth=-4, readOnly=False)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "consent" ---
    # create an object to store info about Routine consent
    consent = data.Routine(
        name='consent',
        components=[consentform, button],
    )
    consent.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset button to account for continued clicks & clear times on/off
    button.reset()
    # store start times for consent
    consent.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    consent.tStart = globalClock.getTime(format='float')
    consent.status = STARTED
    thisExp.addData('consent.started', consent.tStart)
    consent.maxDuration = None
    # keep track of which components have finished
    consentComponents = consent.components
    for thisComponent in consent.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "consent" ---
    consent.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *consentform* updates
        
        # if consentform is starting this frame...
        if consentform.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            consentform.frameNStart = frameN  # exact frame index
            consentform.tStart = t  # local t and not account for scr refresh
            consentform.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(consentform, 'tStartRefresh')  # time at next scr refresh
            # update status
            consentform.status = STARTED
            consentform.setAutoDraw(True)
        
        # if consentform is active this frame...
        if consentform.status == STARTED:
            # update params
            pass
        # *button* updates
        
        # if button is starting this frame...
        if button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            button.frameNStart = frameN  # exact frame index
            button.tStart = t  # local t and not account for scr refresh
            button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button, 'tStartRefresh')  # time at next scr refresh
            # update status
            button.status = STARTED
            win.callOnFlip(button.buttonClock.reset)
            button.setAutoDraw(True)
        
        # if button is active this frame...
        if button.status == STARTED:
            # update params
            pass
            # check whether button has been pressed
            if button.isClicked:
                if not button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button.timesOn.append(button.buttonClock.getTime())
                    button.timesOff.append(button.buttonClock.getTime())
                elif len(button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button.timesOff[-1] = button.buttonClock.getTime()
                if not button.wasClicked:
                    # end routine when button is clicked
                    continueRoutine = False
                if not button.wasClicked:
                    # run callback code when button is clicked
                    pass
        # take note of whether button was clicked, so that next frame we know if clicks are new
        button.wasClicked = button.isClicked and button.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            consent.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in consent.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "consent" ---
    for thisComponent in consent.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for consent
    consent.tStop = globalClock.getTime(format='float')
    consent.tStopRefresh = tThisFlipGlobal
    thisExp.addData('consent.stopped', consent.tStop)
    thisExp.addData('button.numClicks', button.numClicks)
    if button.numClicks:
       thisExp.addData('button.timesOn', button.timesOn)
       thisExp.addData('button.timesOff', button.timesOff)
    else:
       thisExp.addData('button.timesOn', "")
       thisExp.addData('button.timesOff', "")
    thisExp.nextEntry()
    # the Routine "consent" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('dsr_p.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "dsr_personal" ---
        # create an object to store info about Routine dsr_personal
        dsr_personal = data.Routine(
            name='dsr_personal',
            components=[text, question, slider],
        )
        dsr_personal.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        question.setText(dsr_per)
        slider.reset()
        # store start times for dsr_personal
        dsr_personal.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        dsr_personal.tStart = globalClock.getTime(format='float')
        dsr_personal.status = STARTED
        thisExp.addData('dsr_personal.started', dsr_personal.tStart)
        dsr_personal.maxDuration = None
        # keep track of which components have finished
        dsr_personalComponents = dsr_personal.components
        for thisComponent in dsr_personal.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "dsr_personal" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        dsr_personal.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # *question* updates
            
            # if question is starting this frame...
            if question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question.frameNStart = frameN  # exact frame index
                question.tStart = t  # local t and not account for scr refresh
                question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'question.started')
                # update status
                question.status = STARTED
                question.setAutoDraw(True)
            
            # if question is active this frame...
            if question.status == STARTED:
                # update params
                pass
            
            # *slider* updates
            
            # if slider is starting this frame...
            if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider.frameNStart = frameN  # exact frame index
                slider.tStart = t  # local t and not account for scr refresh
                slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider.started')
                # update status
                slider.status = STARTED
                slider.setAutoDraw(True)
            
            # if slider is active this frame...
            if slider.status == STARTED:
                # update params
                pass
            
            # Check slider for response to end Routine
            if slider.getRating() is not None and slider.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                dsr_personal.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in dsr_personal.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "dsr_personal" ---
        for thisComponent in dsr_personal.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for dsr_personal
        dsr_personal.tStop = globalClock.getTime(format='float')
        dsr_personal.tStopRefresh = tThisFlipGlobal
        thisExp.addData('dsr_personal.stopped', dsr_personal.tStop)
        trials.addData('slider.response', slider.getRating())
        trials.addData('slider.rt', slider.getRT())
        # the Routine "dsr_personal" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler2(
        name='trials_2',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('dsr_e.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "dsr_experience" ---
        # create an object to store info about Routine dsr_experience
        dsr_experience = data.Routine(
            name='dsr_experience',
            components=[text2, question2, slider_2],
        )
        dsr_experience.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        question2.setText(dsr_exp)
        slider_2.reset()
        # store start times for dsr_experience
        dsr_experience.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        dsr_experience.tStart = globalClock.getTime(format='float')
        dsr_experience.status = STARTED
        thisExp.addData('dsr_experience.started', dsr_experience.tStart)
        dsr_experience.maxDuration = None
        # keep track of which components have finished
        dsr_experienceComponents = dsr_experience.components
        for thisComponent in dsr_experience.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "dsr_experience" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        dsr_experience.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text2* updates
            
            # if text2 is starting this frame...
            if text2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text2.frameNStart = frameN  # exact frame index
                text2.tStart = t  # local t and not account for scr refresh
                text2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text2, 'tStartRefresh')  # time at next scr refresh
                # update status
                text2.status = STARTED
                text2.setAutoDraw(True)
            
            # if text2 is active this frame...
            if text2.status == STARTED:
                # update params
                pass
            
            # *question2* updates
            
            # if question2 is starting this frame...
            if question2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question2.frameNStart = frameN  # exact frame index
                question2.tStart = t  # local t and not account for scr refresh
                question2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question2, 'tStartRefresh')  # time at next scr refresh
                # update status
                question2.status = STARTED
                question2.setAutoDraw(True)
            
            # if question2 is active this frame...
            if question2.status == STARTED:
                # update params
                pass
            
            # *slider_2* updates
            
            # if slider_2 is starting this frame...
            if slider_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_2.frameNStart = frameN  # exact frame index
                slider_2.tStart = t  # local t and not account for scr refresh
                slider_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                slider_2.status = STARTED
                slider_2.setAutoDraw(True)
            
            # if slider_2 is active this frame...
            if slider_2.status == STARTED:
                # update params
                pass
            
            # Check slider_2 for response to end Routine
            if slider_2.getRating() is not None and slider_2.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                dsr_experience.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in dsr_experience.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "dsr_experience" ---
        for thisComponent in dsr_experience.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for dsr_experience
        dsr_experience.tStop = globalClock.getTime(format='float')
        dsr_experience.tStopRefresh = tThisFlipGlobal
        thisExp.addData('dsr_experience.stopped', dsr_experience.tStop)
        trials_2.addData('slider_2.response', slider_2.getRating())
        # the Routine "dsr_experience" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_2'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "instr" ---
    # create an object to store info about Routine instr
    instr = data.Routine(
        name='instr',
        components=[instruction, key_resp],
    )
    instr.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for instr
    instr.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr.tStart = globalClock.getTime(format='float')
    instr.status = STARTED
    thisExp.addData('instr.started', instr.tStart)
    instr.maxDuration = None
    # keep track of which components have finished
    instrComponents = instr.components
    for thisComponent in instr.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr" ---
    instr.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction* updates
        
        # if instruction is starting this frame...
        if instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction.frameNStart = frameN  # exact frame index
            instruction.tStart = t  # local t and not account for scr refresh
            instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction, 'tStartRefresh')  # time at next scr refresh
            # update status
            instruction.status = STARTED
            instruction.setAutoDraw(True)
        
        # if instruction is active this frame...
        if instruction.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            key_resp.clock.reset()  # now t=0
            key_resp.clearEvents(eventType='keyboard')
        if key_resp.status == STARTED:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr" ---
    for thisComponent in instr.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr
    instr.tStop = globalClock.getTime(format='float')
    instr.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr.stopped', instr.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_3 = data.TrialHandler2(
        name='trials_3',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('imagestim.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_3)  # add the loop to the experiment
    thisTrial_3 = trials_3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
    if thisTrial_3 != None:
        for paramName in thisTrial_3:
            globals()[paramName] = thisTrial_3[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_3 in trials_3:
        currentLoop = trials_3
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
        if thisTrial_3 != None:
            for paramName in thisTrial_3:
                globals()[paramName] = thisTrial_3[paramName]
        
        # --- Prepare to start Routine "encoding" ---
        # create an object to store info about Routine encoding
        encoding = data.Routine(
            name='encoding',
            components=[fixation, study_image, studyword],
        )
        encoding.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        study_image.setImage(imagefile)
        studyword.setText(study_word)
        # store start times for encoding
        encoding.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        encoding.tStart = globalClock.getTime(format='float')
        encoding.status = STARTED
        thisExp.addData('encoding.started', encoding.tStart)
        encoding.maxDuration = None
        # keep track of which components have finished
        encodingComponents = encoding.components
        for thisComponent in encoding.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "encoding" ---
        # if trial has changed, end Routine now
        if isinstance(trials_3, data.TrialHandler2) and thisTrial_3.thisN != trials_3.thisTrial.thisN:
            continueRoutine = False
        encoding.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *study_image* updates
            
            # if study_image is starting this frame...
            if study_image.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                # keep track of start time/frame for later
                study_image.frameNStart = frameN  # exact frame index
                study_image.tStart = t  # local t and not account for scr refresh
                study_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(study_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'study_image.started')
                # update status
                study_image.status = STARTED
                study_image.setAutoDraw(True)
            
            # if study_image is active this frame...
            if study_image.status == STARTED:
                # update params
                pass
            
            # if study_image is stopping this frame...
            if study_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > study_image.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    study_image.tStop = t  # not accounting for scr refresh
                    study_image.tStopRefresh = tThisFlipGlobal  # on global time
                    study_image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'study_image.stopped')
                    # update status
                    study_image.status = FINISHED
                    study_image.setAutoDraw(False)
            
            # *studyword* updates
            
            # if studyword is starting this frame...
            if studyword.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                # keep track of start time/frame for later
                studyword.frameNStart = frameN  # exact frame index
                studyword.tStart = t  # local t and not account for scr refresh
                studyword.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(studyword, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'studyword.started')
                # update status
                studyword.status = STARTED
                studyword.setAutoDraw(True)
            
            # if studyword is active this frame...
            if studyword.status == STARTED:
                # update params
                pass
            
            # if studyword is stopping this frame...
            if studyword.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > studyword.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    studyword.tStop = t  # not accounting for scr refresh
                    studyword.tStopRefresh = tThisFlipGlobal  # on global time
                    studyword.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'studyword.stopped')
                    # update status
                    studyword.status = FINISHED
                    studyword.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                encoding.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in encoding.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "encoding" ---
        for thisComponent in encoding.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for encoding
        encoding.tStop = globalClock.getTime(format='float')
        encoding.tStopRefresh = tThisFlipGlobal
        thisExp.addData('encoding.stopped', encoding.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if encoding.maxDurationReached:
            routineTimer.addTime(-encoding.maxDuration)
        elif encoding.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_3'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "test_instruction" ---
    # create an object to store info about Routine test_instruction
    test_instruction = data.Routine(
        name='test_instruction',
        components=[test_instr, key_resp_2],
    )
    test_instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for test_instruction
    test_instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    test_instruction.tStart = globalClock.getTime(format='float')
    test_instruction.status = STARTED
    thisExp.addData('test_instruction.started', test_instruction.tStart)
    test_instruction.maxDuration = None
    # keep track of which components have finished
    test_instructionComponents = test_instruction.components
    for thisComponent in test_instruction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "test_instruction" ---
    test_instruction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *test_instr* updates
        
        # if test_instr is starting this frame...
        if test_instr.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
            # keep track of start time/frame for later
            test_instr.frameNStart = frameN  # exact frame index
            test_instr.tStart = t  # local t and not account for scr refresh
            test_instr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(test_instr, 'tStartRefresh')  # time at next scr refresh
            # update status
            test_instr.status = STARTED
            test_instr.setAutoDraw(True)
        
        # if test_instr is active this frame...
        if test_instr.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and t >= 2.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('key_resp_2.started', t)
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            key_resp_2.clock.reset()  # now t=0
            key_resp_2.clearEvents(eventType='keyboard')
        if key_resp_2.status == STARTED:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            test_instruction.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in test_instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "test_instruction" ---
    for thisComponent in test_instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for test_instruction
    test_instruction.tStop = globalClock.getTime(format='float')
    test_instruction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('test_instruction.stopped', test_instruction.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "test_instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_4 = data.TrialHandler2(
        name='trials_4',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('wordtest.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_4)  # add the loop to the experiment
    thisTrial_4 = trials_4.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
    if thisTrial_4 != None:
        for paramName in thisTrial_4:
            globals()[paramName] = thisTrial_4[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_4 in trials_4:
        currentLoop = trials_4
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
        if thisTrial_4 != None:
            for paramName in thisTrial_4:
                globals()[paramName] = thisTrial_4[paramName]
        
        # --- Prepare to start Routine "testing" ---
        # create an object to store info about Routine testing
        testing = data.Routine(
            name='testing',
            components=[t_word, slider_3],
        )
        testing.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        t_word.setText(test_word)
        slider_3.reset()
        # store start times for testing
        testing.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        testing.tStart = globalClock.getTime(format='float')
        testing.status = STARTED
        thisExp.addData('testing.started', testing.tStart)
        testing.maxDuration = None
        # keep track of which components have finished
        testingComponents = testing.components
        for thisComponent in testing.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "testing" ---
        # if trial has changed, end Routine now
        if isinstance(trials_4, data.TrialHandler2) and thisTrial_4.thisN != trials_4.thisTrial.thisN:
            continueRoutine = False
        testing.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *t_word* updates
            
            # if t_word is starting this frame...
            if t_word.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_word.frameNStart = frameN  # exact frame index
                t_word.tStart = t  # local t and not account for scr refresh
                t_word.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_word, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_word.started')
                # update status
                t_word.status = STARTED
                t_word.setAutoDraw(True)
            
            # if t_word is active this frame...
            if t_word.status == STARTED:
                # update params
                pass
            
            # *slider_3* updates
            
            # if slider_3 is starting this frame...
            if slider_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_3.frameNStart = frameN  # exact frame index
                slider_3.tStart = t  # local t and not account for scr refresh
                slider_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_3.started')
                # update status
                slider_3.status = STARTED
                slider_3.setAutoDraw(True)
            
            # if slider_3 is active this frame...
            if slider_3.status == STARTED:
                # update params
                pass
            
            # Check slider_3 for response to end Routine
            if slider_3.getRating() is not None and slider_3.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                testing.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in testing.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "testing" ---
        for thisComponent in testing.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for testing
        testing.tStop = globalClock.getTime(format='float')
        testing.tStopRefresh = tThisFlipGlobal
        thisExp.addData('testing.stopped', testing.tStop)
        trials_4.addData('slider_3.response', slider_3.getRating())
        trials_4.addData('slider_3.rt', slider_3.getRT())
        # the Routine "testing" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_4'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[thanks, button_2],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset button_2 to account for continued clicks & clear times on/off
    button_2.reset()
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thanks* updates
        
        # if thanks is starting this frame...
        if thanks.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thanks.frameNStart = frameN  # exact frame index
            thanks.tStart = t  # local t and not account for scr refresh
            thanks.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thanks, 'tStartRefresh')  # time at next scr refresh
            # update status
            thanks.status = STARTED
            thanks.setAutoDraw(True)
        
        # if thanks is active this frame...
        if thanks.status == STARTED:
            # update params
            pass
        # *button_2* updates
        
        # if button_2 is starting this frame...
        if button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            button_2.frameNStart = frameN  # exact frame index
            button_2.tStart = t  # local t and not account for scr refresh
            button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            button_2.status = STARTED
            win.callOnFlip(button_2.buttonClock.reset)
            button_2.setAutoDraw(True)
        
        # if button_2 is active this frame...
        if button_2.status == STARTED:
            # update params
            pass
            # check whether button_2 has been pressed
            if button_2.isClicked:
                if not button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button_2.timesOn.append(button_2.buttonClock.getTime())
                    button_2.timesOff.append(button_2.buttonClock.getTime())
                elif len(button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button_2.timesOff[-1] = button_2.buttonClock.getTime()
                if not button_2.wasClicked:
                    # end routine when button_2 is clicked
                    continueRoutine = False
                if not button_2.wasClicked:
                    # run callback code when button_2 is clicked
                    pass
        # take note of whether button_2 was clicked, so that next frame we know if clicks are new
        button_2.wasClicked = button_2.isClicked and button_2.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    thisExp.addData('button_2.numClicks', button_2.numClicks)
    if button_2.numClicks:
       thisExp.addData('button_2.timesOn', button_2.timesOn)
       thisExp.addData('button_2.timesOff', button_2.timesOff)
    else:
       thisExp.addData('button_2.timesOn', "")
       thisExp.addData('button_2.timesOff', "")
    thisExp.nextEntry()
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "brk" ---
    # create an object to store info about Routine brk
    brk = data.Routine(
        name='brk',
        components=[text_4, key_resp_3],
    )
    brk.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # store start times for brk
    brk.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    brk.tStart = globalClock.getTime(format='float')
    brk.status = STARTED
    thisExp.addData('brk.started', brk.tStart)
    brk.maxDuration = None
    # keep track of which components have finished
    brkComponents = brk.components
    for thisComponent in brk.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "brk" ---
    brk.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            key_resp_3.clock.reset()  # now t=0
            key_resp_3.clearEvents(eventType='keyboard')
        if key_resp_3.status == STARTED:
            theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            brk.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in brk.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "brk" ---
    for thisComponent in brk.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for brk
    brk.tStop = globalClock.getTime(format='float')
    brk.tStopRefresh = tThisFlipGlobal
    thisExp.addData('brk.stopped', brk.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "brk" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "lsas_instr" ---
    # create an object to store info about Routine lsas_instr
    lsas_instr = data.Routine(
        name='lsas_instr',
        components=[text_5, key_resp_4],
    )
    lsas_instr.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_4
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # store start times for lsas_instr
    lsas_instr.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    lsas_instr.tStart = globalClock.getTime(format='float')
    lsas_instr.status = STARTED
    thisExp.addData('lsas_instr.started', lsas_instr.tStart)
    lsas_instr.maxDuration = None
    # keep track of which components have finished
    lsas_instrComponents = lsas_instr.components
    for thisComponent in lsas_instr.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "lsas_instr" ---
    lsas_instr.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_5* updates
        
        # if text_5 is starting this frame...
        if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_5.frameNStart = frameN  # exact frame index
            text_5.tStart = t  # local t and not account for scr refresh
            text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_5.status = STARTED
            text_5.setAutoDraw(True)
        
        # if text_5 is active this frame...
        if text_5.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            key_resp_4.clock.reset()  # now t=0
            key_resp_4.clearEvents(eventType='keyboard')
        if key_resp_4.status == STARTED:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            lsas_instr.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in lsas_instr.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "lsas_instr" ---
    for thisComponent in lsas_instr.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for lsas_instr
    lsas_instr.tStop = globalClock.getTime(format='float')
    lsas_instr.tStopRefresh = tThisFlipGlobal
    thisExp.addData('lsas_instr.stopped', lsas_instr.tStop)
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "lsas_instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_5 = data.TrialHandler2(
        name='trials_5',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('lsas_survey.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_5)  # add the loop to the experiment
    thisTrial_5 = trials_5.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_5.rgb)
    if thisTrial_5 != None:
        for paramName in thisTrial_5:
            globals()[paramName] = thisTrial_5[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_5 in trials_5:
        currentLoop = trials_5
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_5.rgb)
        if thisTrial_5 != None:
            for paramName in thisTrial_5:
                globals()[paramName] = thisTrial_5[paramName]
        
        # --- Prepare to start Routine "lsas" ---
        # create an object to store info about Routine lsas
        lsas = data.Routine(
            name='lsas',
            components=[question_3, text_2, slider_4, text_3, slider_5],
        )
        lsas.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        question_3.setText(lsas_situation)
        slider_4.reset()
        slider_5.reset()
        # store start times for lsas
        lsas.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        lsas.tStart = globalClock.getTime(format='float')
        lsas.status = STARTED
        thisExp.addData('lsas.started', lsas.tStart)
        lsas.maxDuration = None
        # keep track of which components have finished
        lsasComponents = lsas.components
        for thisComponent in lsas.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "lsas" ---
        # if trial has changed, end Routine now
        if isinstance(trials_5, data.TrialHandler2) and thisTrial_5.thisN != trials_5.thisTrial.thisN:
            continueRoutine = False
        lsas.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *question_3* updates
            
            # if question_3 is starting this frame...
            if question_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                question_3.frameNStart = frameN  # exact frame index
                question_3.tStart = t  # local t and not account for scr refresh
                question_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(question_3, 'tStartRefresh')  # time at next scr refresh
                # update status
                question_3.status = STARTED
                question_3.setAutoDraw(True)
            
            # if question_3 is active this frame...
            if question_3.status == STARTED:
                # update params
                pass
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # *slider_4* updates
            
            # if slider_4 is starting this frame...
            if slider_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_4.frameNStart = frameN  # exact frame index
                slider_4.tStart = t  # local t and not account for scr refresh
                slider_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_4, 'tStartRefresh')  # time at next scr refresh
                # update status
                slider_4.status = STARTED
                slider_4.setAutoDraw(True)
            
            # if slider_4 is active this frame...
            if slider_4.status == STARTED:
                # update params
                pass
            
            # *text_3* updates
            
            # if text_3 is starting this frame...
            if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.started')
                # update status
                text_3.status = STARTED
                text_3.setAutoDraw(True)
            
            # if text_3 is active this frame...
            if text_3.status == STARTED:
                # update params
                pass
            
            # *slider_5* updates
            
            # if slider_5 is starting this frame...
            if slider_5.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_5.frameNStart = frameN  # exact frame index
                slider_5.tStart = t  # local t and not account for scr refresh
                slider_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                slider_5.status = STARTED
                slider_5.setAutoDraw(True)
            
            # if slider_5 is active this frame...
            if slider_5.status == STARTED:
                # update params
                pass
            
            # Check slider_5 for response to end Routine
            if slider_5.getRating() is not None and slider_5.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                lsas.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in lsas.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "lsas" ---
        for thisComponent in lsas.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for lsas
        lsas.tStop = globalClock.getTime(format='float')
        lsas.tStopRefresh = tThisFlipGlobal
        thisExp.addData('lsas.stopped', lsas.tStop)
        trials_5.addData('slider_4.response', slider_4.getRating())
        trials_5.addData('slider_5.response', slider_5.getRating())
        # the Routine "lsas" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_5'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
