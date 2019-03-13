<!DOCTYPE html>
<html>
<head>
<?php
header('Cache-control: no-cache');
?>
<script>
function httpGet(theUrl)
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open( "GET", theUrl, false );
    xmlHttp.send( null );
    return xmlHttp.responseText;
}
window.setInterval(function()
{
    document.getElementById('mlcam').src = "detectimgs/RD/driveway.jpg?random="+new Date().getTime();
    document.getElementById('mlcam2').src = "detectimgs/RD/frontyard.jpg?random="+new Date().getTime();
}, 450);
</script>
<style>
body {
    background-color: black;
}
</style>
</head>
<body>
<center>
	<img id="mlcam" src="detectimgs/RD/drivewaytmp.jpg"
		onError="this.onerror=null;this.src='detectimgs/RD/driveway.jpg';"
	>
<div style="line-height:40%;">
    <br>
</div>
	<img id="mlcam2" src="detectimgs/RD/frontyardtmp.jpg"
		onError="this.onerror=null;this.src='detectimgs/RD/frontyard.jpg';"
	>
</center>
</body>
</html>
