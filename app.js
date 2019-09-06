var webdnn = require('webdnn');
var nj = require('numjs');
var runner = null;

async function test(num) {
    runner = await webdnn.load('model');

    let x = runner.inputs[0];
    let y = runner.outputs[0];

    //ラベル付きガウス分布生成
    let array = nj.random(100);
    let label = nj.zeros(10);
    label.set(num, 1);
    init_x = nj.concatenate([array, label]);
    x.set(init_x.tolist());

    await runner.run();

    var canvas = document.getElementById('output');
    webdnn.Image.setImageArrayToCanvas(y.toActual(), 28, 28, document.getElementById('output'), {
            dstW: canvas.getAttribute('width'),
            dstH: canvas.getAttribute('height'),
            scale: [255, 255, 255],
            bias: [0, 0, 0],
            color: webdnn.Image.Color.GREY,
            order: webdnn.Image.Order.CHW //(色、縦、横)
        });

}

window.onload = function() {
    document.getElementById('button').onclick = function() {
        let num = document.getElementById('number').value;
        console.log(num);
        test(num);
    }
}