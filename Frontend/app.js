const express = require('express');
const WEB_PORT = 3000;
const app = express()

// const server = express()
// 	.listen(WS_PORT, () => { console.log(`WEB: Listening on ${WS_PORT}`) });


// const SocketServer = require('ws').Server
// const WS_PORT = 3001;
// const wss = new SocketServer({ server });

// wss.on('connection', ws => {

//     console.log('Client connected')

//     ws.on('close', () => {
//         console.log('Close connected')
//     })
// });


app.get('/', (req, res) => {
	res.sendFile(__dirname + '/index.html');
})

app.get('/style.css', (req, res) => {
	res.sendFile(__dirname + '/style.css');
})

app.get('/index.js', (req, res) => {
	res.sendFile(__dirname + '/index.js');
})

app.get('/favicon.ico', (req, res) => {
	res.sendFile(__dirname + '/favicon.ico');
})

app.listen(WEB_PORT, () => { console.log(`WEB: Listening on ${WEB_PORT}`) });



const fs = require('fs');

app.get('/videoplayer', (req, res) => {
	const range = req.headers.range
	const videoPath = './test.mp4';
	const videoSize = fs.statSync(videoPath).size
	const chunkSize = 1 * 1e6;
	const start = Number(range.replace(/\D/g, ""))
	const end = Math.min(start + chunkSize, videoSize - 1)
	const contentLength = end - start + 1;

	const headers = {
		"Content-Range": `bytes ${start}-${end}/${videoSize}`,
		"Accept-Ranges": "bytes",
		"Content-Length": contentLength,
		"Content-Type": "video/mp4"
	}

	res.writeHead(206, headers)
	const stream = fs.createReadStream(videoPath, {
		start,
		end
	})
	stream.pipe(res)
})


