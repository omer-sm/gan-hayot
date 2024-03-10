// electron.js
import { app, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import isDev from 'electron-is-dev'

let mainWindow;
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1080,
    height: 720,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      fullscreen: true,
    },
  });
  //mainWindow.removeMenu()
  const startURL = isDev
    ? 'http://localhost:3000'
    : `file://${join(__dirname, '../build/index.html')}`;

  mainWindow.loadURL(startURL);

  mainWindow.on('closed', () => (mainWindow = null));
}
ipcMain.on('call-py', (event, arg) => {
  // Execute your function here
  console.log(arg)
});


app.on('ready', createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});