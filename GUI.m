% Developed by Vikram Bajaj
function varargout = GUI(varargin)
% GUI MATLAB code for GUI.fig
%      GUI, by itself, creates a new GUI or raises the existing
%      singleton*.
%
%      H = GUI returns the handle to a new GUI or the handle to
%      the existing singleton*.
%
%      GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI.M with the given input arguments.
%
%      GUI('Property','Value',...) creates a new GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUI

% Last Modified by GUIDE v2.5 30-Sep-2016 20:10:29

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GUI is made visible.
function GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUI (see VARARGIN)

% Choose default command line output for GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);
set(handles.uploadbutton,'visible','off');
set(handles.capturebutton,'visible','off');
set(handles.preview,'XtickLabel',[],'YtickLabel',[]);

% --- Outputs from this function are returned to the command line.
function varargout = GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in radiobutton1.
function radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton1
set(handles.uploadbutton,'visible','off');
set(handles.capturebutton,'visible','on');
set(handles.emotion,'string',''); % clear result, if any
% Start webcam preview
handles.vid = videoinput('winvideo',1); % for Integrated Webcam
% handles.vid = videoinput('winvideo',1, 'I420_1280x960'); % for external webcam
% only capture one frame per trigger
handles.vid.FramesPerTrigger = 1;
% output would image in RGB color space
handles.vid.ReturnedColorspace = 'rgb';
% tell MATLAB to start the webcam on user request, not automatically
triggerconfig(handles.vid, 'manual');
% we need this to know the image height and width
vidRes = get(handles.vid, 'VideoResolution');
% image width
imWidth = vidRes(1);
% image height
imHeight = vidRes(2);
% number of bands of our image (should be 3 because it's RGB)
nBands = get(handles.vid, 'NumberOfBands');
% create an empty image container and show it on axPreview
hImage = image(zeros(imHeight, imWidth, nBands), 'parent', handles.preview);
% begin the webcam preview
try
delete('testImage.jpg');
catch
end
preview(handles.vid, hImage);
guidata(hObject,handles); 

% --- Executes on button press in radiobutton2.
function radiobutton2_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton2
set(handles.uploadbutton,'visible','on');
set(handles.capturebutton,'visible','off');
set(handles.emotion,'string',''); % clear result, if any

function emotion_Callback(hObject, eventdata, handles)
% hObject    handle to emotion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of emotion as text
%        str2double(get(hObject,'String')) returns contents of emotion as a double


% --- Executes during object creation, after setting all properties.
function emotion_CreateFcn(hObject, eventdata, handles)
% hObject    handle to emotion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in go.
function go_Callback(hObject, eventdata, handles)
% hObject    handle to go (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load('categoryClassifier.mat');
% load('bag.mat');
img = imread('testImage.jpg');
[labelIdx, scores] = predict(categoryClassifier, img);
label_names = {'Angry','Happy', 'Neutral', 'Sad'};
C1=label_names{labelIdx};
set(handles.emotion,'string',C1);

% --- Executes on button press in uploadbutton.
function uploadbutton_Callback(hObject, eventdata, handles)
% hObject    handle to uploadbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
FileName='';
[filename, foldername] = uigetfile({'*.jpg'}, 'Select Image');
if filename ~= 0
    FileName = fullfile(foldername, filename);
end
% % % save uploaded image in testImage.jpg and show confirmation dialog
FileName;
F = imread(FileName);
imwrite(F, 'testImage.jpg');  %%% the image is saved by this name in the pwd
% show image on axes
imshow(F, 'Parent', handles.preview);
%show a capture confirmation message
h = msgbox(sprintf('\nImage Uploaded.\n\nClick Go!'));


% --- Executes on button press in reset.
function reset_Callback(hObject, eventdata, handles)
% hObject    handle to reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cla; % clear axes
set(handles.emotion,'string',''); % clear result


% --- Executes on button press in capturebutton.
function capturebutton_Callback(hObject, eventdata, handles)
% hObject    handle to capturebutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% prepare for capturing the image preview
start(handles.vid);
% pause for 3 seconds to give the webcam a "warm-up" time
pause(3);
% capture image
trigger(handles.vid);
stoppreview(handles.vid);
% get the captured image data and save it in capt variable
capt = getdata(handles.vid);
handles.image = capt;
imwrite(capt, 'testImage.jpg');  %%% the image is saved by this name in the pwd
handles.preview;
%show a capture confirmation message
h = msgbox(sprintf('\nImage Captured!\n\nClick Go!'));



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
