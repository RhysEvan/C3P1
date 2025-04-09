const {FusesPlugin} = require('@electron-forge/plugin-fuses');
const {FuseV1Options, FuseVersion} = require('@electron/fuses');
const path = require('path');
const fs = require('fs');
const {encryptFile} = require('./scripts/encrypt-python');

module.exports = {
    packagerConfig: {
        asar: true,
        extraResource: [
            './src/settings.yaml',
            './scripts/Test.py.enc',
        ]
    },
    hooks: {
        packageAfterCopy: async (config, buildPath, electronVersion, platform, arch) => {
            console.log('Encrypting Python script for packaging...');

            const pythonScriptPath = path.join(__dirname, 'scripts', 'Test.py');

            const encryptedSourcePath = path.join(__dirname, 'scripts', 'Test.py.enc');
            const encryptedBuildPath = path.join(buildPath, 'scripts', 'Test.py.enc');

            const success = encryptFile(pythonScriptPath, encryptedSourcePath);

            if (success) {
                const buildScriptsDir = path.join(buildPath, 'scripts');
                if (!fs.existsSync(buildScriptsDir)) {
                    fs.mkdirSync(buildScriptsDir, {recursive: true});
                }

                fs.copyFileSync(encryptedSourcePath, encryptedBuildPath);
                console.log('Python script encrypted and copied to build directory');
            } else {
                console.error('Failed to encrypt Python script');
                process.exit(1);
            }
        }
    },
    rebuildConfig: {},
    makers: [
        {
            name: '@electron-forge/maker-squirrel',
            config: {},
        },
        {
            name: '@electron-forge/maker-zip',
            platforms: ['darwin'],
        },
        {
            name: '@electron-forge/maker-deb',
            config: {},
        },
        {
            name: '@electron-forge/maker-rpm',
            config: {},
        },
    ],
    plugins: [
        {
            name: '@electron-forge/plugin-auto-unpack-natives',
            config: {},
        },
        new FusesPlugin({
            version: FuseVersion.V1,
            [FuseV1Options.RunAsNode]: false,
            [FuseV1Options.EnableCookieEncryption]: true,
            [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
            [FuseV1Options.EnableNodeCliInspectArguments]: false,
            [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true,
            [FuseV1Options.OnlyLoadAppFromAsar]: true,
        }),
    ],
};