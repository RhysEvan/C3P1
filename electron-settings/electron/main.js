const {app, BrowserWindow, ipcMain} = require('electron');
const path = require('path');
const fs = require('fs');
const yaml = require('js-yaml');

// Change path to src directory as requested
const settingsPath = path.join(__dirname, '../src/settings.yaml');

// Function to read settings
function readSettings() {
    try {
        if (fs.existsSync(settingsPath)) {
            const fileContents = fs.readFileSync(settingsPath, 'utf8');
            return yaml.load(fileContents) || {};
        }
        return {};
    } catch (error) {
        console.error('Error reading settings:', error);
        return {};
    }
}

// Function to save settings
function saveSettings(settings) {
    try {
        const yamlStr = yaml.dump(settings);
        fs.writeFileSync(settingsPath, yamlStr, 'utf8');
        return true;
    } catch (error) {
        console.error('Error saving settings:', error);
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
    if (process.platform !== 'darwin') app.quit();
});