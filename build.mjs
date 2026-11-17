import {mkdirSync,copyFileSync} from 'node:fs';
mkdirSync('dist',{recursive:true});
for(const file of ['index.html','style.css','curriculum.js','app.js'])copyFileSync(file,`dist/${file}`);
console.log('Static production build ready in dist/');
