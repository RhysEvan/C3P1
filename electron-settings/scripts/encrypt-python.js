const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// Simple encryption since safeStorage isn't available at build time
function encryptFile(inputPath, outputPath) {
  try {
    console.log(`Encrypting: ${inputPath}`);
    console.log(`Output: ${outputPath}`);

    // Read the source file
    if (!fs.existsSync(inputPath)) {
      console.error(`Source file does not exist: ${inputPath}`);
      process.exit(1);
    }

    const fileContent = fs.readFileSync(inputPath, 'utf8');

    // Generate encryption key and IV
    const key = crypto.randomBytes(32);
    const iv = crypto.randomBytes(16);

    // Encrypt the content
    const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
    let encrypted = cipher.update(fileContent, 'utf8', 'hex');
    encrypted += cipher.final('hex');

    // Store the encryption data
    const encryptedData = {
      key: key.toString('hex'),
      iv: iv.toString('hex'),
      content: encrypted
    };

    // Create output directory if it doesn't exist
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // Write the encrypted data
    fs.writeFileSync(outputPath, JSON.stringify(encryptedData));
    console.log('Encryption successful');

    return true;
  } catch (error) {
    console.error('Encryption failed:', error);
    return false;
  }
}

// When run directly from command line
if (require.main === module) {
  if (process.argv.length < 4) {
    console.error('Usage: node encrypt-python.js <input-file> <output-file>');
    process.exit(1);
  }

  const inputPath = process.argv[2];
  const outputPath = process.argv[3];

  const success = encryptFile(inputPath, outputPath);
  process.exit(success ? 0 : 1);
}

module.exports = { encryptFile };