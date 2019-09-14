var webdnn = require('webdnn');
var nj = require('numjs');
var runner = null;

async function predict(num) {
    runner = await webdnn.load('model');

    let x = runner.inputs[0];
    let y = runner.outputs[0];

    //generate random normal distribution (z-noise)
    let array = nj.random(100);

    //convolution
    let label = nj.zeros(10);
    label.set(num, 1);
    init_x = nj.concatenate([array, label]);

    x.set(init_x.tolist());
    await runner.run();

    //draw
    var canvas = document.getElementById('output');
    webdnn.Image.setImageArrayToCanvas(y.toActual(), 28, 28, document.getElementById('output'), {
            dstW: canvas.getAttribute('width'),
            dstH: canvas.getAttribute('height'),
            scale: [255, 255, 255],
            bias: [0, 0, 0],
            color: webdnn.Image.Color.GREY,
            order: webdnn.Image.Order.CHW
        });

}

window.onload = function() {
    document.getElementById('button').onclick = function() {
        let num = document.getElementById('number').value;
        //console.log(num);
        predict(num);
    }
}