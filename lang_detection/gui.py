import PySimpleGUI as sg

from lang_detection.detection import load_detector

layout = [
    [sg.Text('Enter text to detect language:')],
    [sg.Multiline(size=(None, 10), key='text')],
    [sg.Button('Detect', key='detect')],
    [sg.Text('Detected language:'), sg.Text('', key='lang', size=(10, 1))],
    [sg.FileBrowse('Select model', key='model', file_types=(('JSON', '*.json'),))],
    [sg.Button('Reload model', key='reload')],
    [sg.Text('Model:'), sg.Text('', key='model_name')],
]

window = sg.Window('Language detection', layout)

detector = None

if __name__ == '__main__':
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == 'detect':
            if detector is None:
                sg.popup('Select model first!')
                continue

            detected_lang, score = detector.prompt(values['text'])
            window['lang'].update(detected_lang)
        if event == 'reload':
            if not values['model']:
                sg.popup('Select model first!')
                continue

            detector = load_detector(values['model'])
            window['model_name'].update(values['model'])
