var start = 0;

window.addEventListener('mousedown',function(event){
    if(start == 0)
    start = 1;
    else
    start = 0;
});

var canvas = document.querySelector('canvas');
var ctx = canvas.getContext('2d');

canvas.width = 280;
canvas.height = 280;
canvas.style = "position:absolute; left: 50%; margin-left: -200px"; //canvas is in the center

var mouse = {
    x:undefined,
    y:undefined
}
// get the mouse position
window.addEventListener('mousemove',function(event){
    mouse.x = event.x;
    mouse.y = event.y;
    console.log(mouse);
    if(start == 1)
    {
    ctx.beginPath();
    ctx.arc(mouse.x-480,mouse.y,15,0,Math.PI*2,false);//offset of 480 selected by trial and error
    ctx.strokeStyle = '#A1C181';//'#233D4D';//'#A1C181';
    ctx.fillStyle = '#A1C181';//'#233D4D';//'#A1C181';
    ctx.fill();
    ctx.stroke();
    }
});

function done(){
    console.log('all done!!');
    var canvas = document.querySelector('canvas');
    var imags = canvas.toDataURL("image/png");//.replace("image/png", "image/octet-stream"); //conversion is done to save the image
    //imags=img.replace("image/png", "image/octet-stream");
    console.log(imags);
//To convert the image to png and save it
    var link = document.createElement('a');
    link.href=imags;
    link.download='image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
 //var image = new Image();
/* drawing circle
function Circle(x ,y ){
    this.x = x;
    this.y = y;

}*/
