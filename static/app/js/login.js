/**
 * Created by Tadej on 30.6.2016.
 */

var desiredAnswer
$(document).ready(function () {
    init();
})

function init(){
    var a = Math.floor((Math.random() * 10) + 1);
    var b = Math.floor((Math.random() * 10) + 1);
    $("#answerLabel").text("What is " + a + " * " + b);
    desiredAnswer = a*b
    $('#answerBtn').click(checkAnswer);
}

function checkAnswer(){
    var answer = $("#answer").val()
    if(parseInt(answer, 10) == desiredAnswer){
        window.location = "/app/login_as_guest"
    } else {
         $("#instrLabel").text("Wrong answer");
    }
}

