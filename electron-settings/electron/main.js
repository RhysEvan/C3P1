const {app, BrowserWindow, ipcMain} = require('electron');
const path = require('path');
const fs = require('fs');
const yaml = require('js-yaml');

function getSettingsPath() {
  // In development
  if (process.env.NODE_ENV === 'development') {
    return path.join(__dirname, 'settings.yaml');
  }

  // In production
  return process.env.ELECTRON_RUN_AS_NODE
    ? path.join(__dirname, '../resources/settings.yaml')
    : path.join(process.resourcesPath, 'settings.yaml');
}

function readSettings() {
  try {
    const settingsPath = getSettingsPath();
    if (fs.existsSync(settingsPath)) {
      const fileContents = fs.readFileSync(settingsPath, 'utf8');
      return yaml.load(fileContents) || {};
    }
    return {};
  } catch (error) {
    console.error('Failed to load settings:', error);
    return {};
  }
}

function saveSettings(settings) {
  try {
    const settingsPath = getSettingsPath();
    const yamlStr = yaml.dump(settings);
    fs.writeFileSync(settingsPath, yamlStr, 'utf8');
    return true;
  } catch (error) {
    console.error('Failed to save settings:', error);
    return false;
  }
}

function createWindow() {
    // Your existing createWindow code
    const mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false
        }
    });

    mainWindow.loadFile(path.join(__dirname, '../src/index.html'));
}

// Set up IPC handlers
ipcMain.handle('get-settings', () => {
    return readSettings();
});

ipcMain.handle('save-settings', (_, newSettings) => {
    return saveSettings(newSettings);
});

ipcMain.on('show-settings', (event) => {
    // Implement the logic to show settings
});

// Rest of your existing code
app.whenReady().then(() => {
    const userDataPath = app.getPath('userData');
    app.setPath('userData', path.join(userDataPath, 'electron-cache'));
    createWindow();

    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', function () {
    // On macOS, it's common to keep the app open even when all windows are closed
    if (process.platform !== 'darwin') app.quit();
});