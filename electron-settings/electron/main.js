const {app, BrowserWindow, ipcMain, safeStorage} = require('electron');
const path = require('path');
const fs = require('fs');
const yaml = require('js-yaml');
const {spawn} = require('child_process');
const os = require('os');
const crypto = require('crypto');

// This function will only be used in development for testing
function encryptPythonScript() {
    if (process.env.NODE_ENV === 'development') {
        const scriptPath = path.join(__dirname, '../scripts/Test.py');
        const encryptedPath = path.join(__dirname, '../scripts/Test.py.enc');

        try {
            // Only encrypt if the encrypted file doesn't exist yet
            if (!fs.existsSync(encryptedPath) && fs.existsSync(scriptPath)) {
                console.log('Encrypting Python script for development testing');
                const scriptContent = fs.readFileSync(scriptPath, 'utf8');
                const encryptBuffer = safeStorage.encryptString(scriptContent);
                fs.writeFileSync(encryptedPath, encryptBuffer);
                console.log('Python script encrypted successfully.');
            }
        } catch (error) {
            console.error('Error encrypting Python script:', error);
        }
    }
}

function decryptPythonScript(tempPath) {
    try {
        const crypto = require('crypto');

        // Path to encrypted file
        const encryptedPath = app.isPackaged
            ? path.join(process.resourcesPath, 'Test.py.enc')
            : path.join(__dirname, '../scripts/Test.py.enc');

        // Read encrypted data
        if (!fs.existsSync(encryptedPath)) {
            throw new Error(`Encrypted file not found: ${encryptedPath}`);
        }

        const encryptedData = JSON.parse(fs.readFileSync(encryptedPath, 'utf8'));

        // Get key and IV
        const key = Buffer.from(encryptedData.key, 'hex');
        const iv = Buffer.from(encryptedData.iv, 'hex');

        // Decrypt
        const decipher = crypto.createDecipheriv('aes-256-cbc', key, iv);
        let decrypted = decipher.update(encryptedData.content, 'hex', 'utf8');
        decrypted += decipher.final('utf8');

        // Write to temp file
        fs.writeFileSync(tempPath, decrypted, 'utf8');
        return true;
    } catch (error) {
        console.error('Failed to decrypt Python script:', error);
        return false;
    }
}

function runPythonScript(args = []) {
    return new Promise((resolve, reject) => {
        const tmpDir = os.tmpdir();
        const tempScriptPath = path.join(tmpDir, `script_${Date.now()}.py`); // Fixed Date.now

        try {
            let success;
            if (!app.isPackaged) {
                const devScriptPath = path.join(__dirname, '../scripts/Test.py');
                fs.copyFileSync(devScriptPath, tempScriptPath);
                success = true;
            } else {
                // In production, decrypt the script
                success = decryptPythonScript(tempScriptPath);
            }

            if (!success) {
                throw new Error("Failed to prepare Python script");
            }

            // Run the Python script
            const pythonProcess = spawn('python', [tempScriptPath, ...args]);

            let output = '';
            let errorOutput = '';

            pythonProcess.stdout.on('data', (data) => {
                output += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });

            pythonProcess.on('close', (code) => {
                // Delete the temporary file
                try {
                    fs.unlinkSync(tempScriptPath);
                } catch (err) {
                    console.error('Failed to delete temporary script:', err);
                }

                if (code === 0) {
                    try {
                        const result = JSON.parse(output);
                        resolve(result);
                    } catch (error) {
                        resolve({status: 'success', raw: output});
                    }
                } else {
                    reject({status: 'error', code, error: errorOutput});
                }
            });
        } catch (error) {
            // Clean up and reject if there's an error
            try {
                if (fs.existsSync(tempScriptPath)) {
                    fs.unlinkSync(tempScriptPath);
                }
            } catch (err) {
                console.error('Failed to delete temporary script after error:', err);
            }
            reject({status: 'error', message: error.toString()});
        }
    });
}

// Python script handler
ipcMain.handle('run-python', async (_, args) => {
    try {
        const jsonArgs = JSON.stringify(args);
        const result = await runPythonScript([jsonArgs]);
        return result;
    } catch (error) {
        console.error('Error running Python script:', error);
        return {status: 'error', message: error.toString()};
    }
});

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
    const mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false
        }
    });

    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
}

// Set up IPC handlers
ipcMain.handle('get-settings', () => {
    return readSettings();
});

ipcMain.handle('save-settings', (_, newSettings) => {
    return saveSettings(newSettings);
});

ipcMain.on('show-settings', () => {
    // Implement the logic to show settings
});

app.whenReady().then(() => {
    if (safeStorage.isEncryptionAvailable()) {
        // Only used in development for testing
        encryptPythonScript();
    } else {
        console.error('Encryption is not available on this platform.');
    }

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