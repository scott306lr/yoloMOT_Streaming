const {
	property,
    select,
    selectAll
} = d3;

const selectRes = select('#selectResolution');
const videoPlay = document.getElementById('mp4Player');
var nowRes = "480p";


var url = 'http://127.0.0.1';
var port = '12345';

var socket = null;

$(document).ready(() => {

	console.log('connecting...')

	socket = io.connect();	
	console.log(socket)

	socket.once('connect', (data) => {
		console.log(data);
	});

	socket.on('play_toggle', (data) => {
		console.log(data);
	});

	socket.on('mode', (data) => {
		console.log(data);
	})
});


function changeResolution() {
	console.log("changeRes " + selectRes.property('value'));

	nowRes = selectRes.property('value');

	if (nowRes == '480p')
		socket.emit('resolution', '480p');
	if (nowRes == '360p')
		socket.emit('resolution', '360p');
	if (nowRes == '720p')
		socket.emit('resolution', '720p');
	if (nowRes == '1080p')
		socket.emit('resolution', '108p');
};

var now_play = true;
function play_pause() {
	console.log(socket)

	if (now_play) {
		select('#playpause').text('Play');
		socket.emit('play_toggle', 'pause');
	}
	else {
		select('#playpause').text('Pause');
		socket.emit('play_toggle', 'play');
	}
	now_play = !now_play;
};

var now_filter = [];
function applyFilter() {
	let s = select('#switchBW').select('input')

	if (s.property('checked')) {
		now_filter = [0];
		socket.emit('mode', 'canny');
	}
	else {
		now_filter = [];
		socket.emit('mode', 'original');
	}

	console.log("applyFilter ", now_filter)
}


var now_conf = 0.5;
var now_classes = [];
function selectClass() {
	now_classes = [];
	for (var i in all_cls) {
		if (all_cls[i].property('checked'))
			now_classes.push(i);
	}
	now_conf = 0.01 * select('#confOutput').property('value');

	socket.emit('show_class_id', now_classes);
	socket.emit('confidence', now_conf);

	console.log('Confidence ', now_conf, ' ,select class ', now_classes);
};

select('#changeRes').on('click', changeResolution);
select('#playpause').on('click', play_pause);
select('#saveClass').on('click', selectClass);
select('#saveFilter').on('click', applyFilter);

let resolutions = ['360p', '480p', '720p', '1080p'];

let slc_res = select('#selectResolution');
for (var i in resolutions) {
	slc_res.append('option')
		.attr('value', resolutions[i])
		.text(resolutions[i]);
}

let classes_dic = {
	0:  'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
	6:  'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
	11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
	16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 
	21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 
	26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 
	31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
	36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 
	41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
	46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
	51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 
	56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 
	61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 
	66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 
	71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 
	76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
};

let slc_cls = select('#selectClass');
var opt_div = null;
var all_cls = [];

for (var i = 0; i < Object.keys(classes_dic).length; i++) {
	if (i % 3 == 0)	
		opt_div = slc_cls.append('div').attr('class', 'cls_row');

	let now_div = opt_div.append('div')
		.attr('class', 'cls_opt')
		.attr('id', 'cls_opt');

	let inp = now_div.append('input')
		.attr('type', 'checkbox')
		.attr('value', i);

	inp.property('checked', true);

	now_div.append('label')
		.attr('class', 'class-label')
		.text(classes_dic[i]);

	all_cls.push(inp);
}
