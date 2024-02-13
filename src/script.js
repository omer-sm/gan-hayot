import fs from "node:fs"

export function r(){
    fs.readFile('./src/test.txt', 'utf8', (err, data) => {
        if (err) {
          console.error(err);
          return;
        }
        console.log(data);
      });
}