import { app, BrowserWindow, dialog, ipcMain } from 'electron';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isDevelopment = process.env.NODE_ENV === 'development';

const getPythonExecutable = () => {
  if (process.env.PYTHON_EXECUTABLE) {
    return process.env.PYTHON_EXECUTABLE;
  }
  if (process.platform === 'win32') {
    return 'python';
  }
  return 'python3';
};

const createWindow = () => {
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    title: 'Clasificador de Propiedades',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  if (isDevelopment) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }

  return mainWindow;
};

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

ipcMain.handle('file:select', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [
      {
        name: 'Datos de propiedades',
        extensions: ['json', 'jsonl', 'csv', 'joblib', 'pkl', 'md', 'markdown'],
      },
    ],
  });

  if (canceled || filePaths.length === 0) {
    return null;
  }

  return filePaths[0];
});

ipcMain.handle('prediction:run', async (_event, filePath) => {
  if (!filePath) {
    return { status: 'error', error: '❌ Error: Debe seleccionar un archivo para procesar' };
  }

  const pythonExecutable = getPythonExecutable();
  const pythonArgs = ['-m', 'property_pricing.src.gui_batch_predict', filePath];

  return new Promise((resolve) => {
    const child = spawn(pythonExecutable, pythonArgs, {
      cwd: path.join(__dirname, '..', '..'),
      env: { ...process.env },
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    child.on('error', (error) => {
      resolve({ status: 'error', error: `❌ Error al ejecutar Python: ${error.message}` });
    });

    child.on('close', () => {
      if (stderr.trim()) {
        resolve({ status: 'error', error: stderr.trim() });
        return;
      }

      try {
        const parsed = JSON.parse(stdout);
        resolve(parsed);
      } catch {
        resolve({
          status: 'error',
          error: '❌ Error: No se pudo interpretar la respuesta del modelo',
          details: stdout,
        });
      }
    });
  });
});
