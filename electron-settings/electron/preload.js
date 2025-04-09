const {contextBridge, ipcRenderer} = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    showSettings: () => ipcRenderer.send('show-settings'),
    getSettings: () => ipcRenderer.invoke('get-settings'),
    saveSettings: (settings) => ipcRenderer.invoke('save-settings', settings),
    runPython: (args) => ipcRenderer.invoke('run-python', args),
});

