import http from 'node:http';
import {readFile} from 'node:fs/promises';
import {resolve,extname} from 'node:path';
const args=process.argv.slice(2),value=(key,fallback)=>args.includes(key)?args[args.indexOf(key)+1]:fallback;
const root=resolve('.'),port=Number(value('--port',3000));
http.createServer(async(req,res)=>{try{const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);const file=resolve(root,'.'+(pathname==='/'?'/index.html':pathname));if(!file.startsWith(root+'/')){res.writeHead(403).end();return}const data=await readFile(file);res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json'}[extname(file)]||'application/octet-stream'));res.end(data)}catch{res.writeHead(404).end('Not found')}}).listen(port,value('--host','127.0.0.1'),()=>console.log(`Serving on ${port}`));
