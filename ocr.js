var start = 0;

window.addEventListener('mousedown',function(event){
    if(start == 0)
    start = 1;
    else
    start = 0;
});

var canvas = document.querySelector('canvas');
var ctx = canvas.getContext('2d');

canvas.width = 600;
canvas.height = 400;
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
    ctx.arc(mouse.x-480,mouse.y,5,0,Math.PI*2,false);
    ctx.fill();
    ctx.stroke();
    }
});

function done(){
    console.log('all done');
    var canvas = document.querySelector('canvas');
    var img = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream"); 
    window.location.href = img;
}
 //var image = new Image();
/* drawing circle
function Circle(x ,y ){
    this.x = x;
    this.y = y;

}*/