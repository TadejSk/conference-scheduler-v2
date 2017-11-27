/**
 * Created by Tadej on 6.7.2015.
 */

function dragStart(e) {
    e.dataTransfer.setData("text/plain", this.href);
}

function addPaper(e) {
    e.preventDefault();
    e.stopPropagation();
    var data = e.dataTransfer.getData("text/plain");
    var ids = data.split('=');
    var id = ids[ids.length - 1];
    var children = this.children;
    var row = -1;
    var col = -1;
    // get day from GET parameters
    var param = location.search
    var day = 0
    if (param != "") {
        day = param.split("=")[1]
    }

    for (var i = 0; i<children.length; i++) {
        if (children[i].tagName == 'INPUT') {
            if (children[i].name == 'row') {
                row = children[i].value;
            }
            if (children[i].name == 'col') {
                col = children[i].value;
            }
        }
    }
    var request = $.ajax({
        method: "POST",
        url: "/app/papers/add_to_schedule/",
        data: {row: row, col: col, id: id, day: day},
    });
    request.done(function (msg) {
        $("#schedule_div").load(location.href + " #schedule_div", function () {
            $(this).children().unwrap();
            init();
        });
        //alert("a")

    });
    request.fail(function (XMLHttpRequest, textStatus, errorThrown) {
        alert("Request failed " + textStatus + errorThrown);
        console.log(XMLHttpRequest.responseText)
    });
}

function removePaper(e) {
    e.preventDefault();
    e.stopPropagation();
    var data = e.dataTransfer.getData("text/plain");
    var ids = data.split('=');
    var id = ids[ids.length - 1];
    var request = $.ajax({
        method: "POST",
        url: "/app/papers/remove_from_schedule/",
        data: {id: id},
    });
    request.done(function (msg) {
        $("#schedule_div").load(location.href + " #schedule_div", function () {
            $(this).children().unwrap();
            init();
        });
    });
    request.fail(function (XMLHttpRequest, textStatus, errorThrown) {
        alert("Request failed " + textStatus + errorThrown);
        console.log(XMLHttpRequest.responseText)
    });
}

function dragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    return false
}

$(document).ready(function () {
    init();
    /*
     $('body').on('dragstart', 'td[name=paper]', dragStart)
     $('body').on('dragover', 'td[name=slot]', dragOver)
     $('body').on('drop', 'td[name=slot]', dragEnd)
     */
})

var lowAnim = false
var highAnim = false
$(document).scroll(function () {
    if ($(this).scrollTop() > 75) {
        if (!lowAnim) {
            lowAnim = true;
            if (highAnim) {
                $('#paper-panel').stop(true, true);
                highAnim = false;
            }
            console.log("low");
            $('#paper-panel').animate({top: '10px'}, 500, "", function () {
                lowAnim = false;
            });
        }
    }
    if ($(this).scrollTop() < 75) {
        if (!highAnim) {
            highAnim = true;
            if (lowAnim) {
                $('#paper-panel').stop(true, true);
                lowAnim = false;
            }
            console.log("high");
            $('#paper-panel').stop(true, true);
            $('#paper-panel').animate({top: '73px'}, 500, "", function () {
                highAnim = false;
            });
        }
    }
});

function init() {
    console.log('a')
    var paper_panel = document.getElementById('paper-panel');
    paper_panel.addEventListener('dragover', dragOver, false);
    paper_panel.addEventListener('drop', removePaper, false);
    var papers = document.getElementsByName('paper');
    for (var i = 0; i<papers.length;i++) {
        papers[i].addEventListener('dragstart', dragStart, false);
    }
    var slots = document.getElementsByName('slot');
    for (var i = 0; i<slots.length;i++) {
        slots[i].addEventListener('dragover', dragOver, false);
        slots[i].addEventListener('drop', addPaper, false);
    }
    var buttons = document.getElementsByName('lock-button');
    for (var i = 0; i<buttons.length; i++){
        buttons[i].addEventListener('click', lock, false);
    }
    /*
    var forms = document.getElementsByName('lock-form');
    for (var form of forms){
        form.addEventListener('submit', doSubmit, false)
    }
    */
};
function lock(e){
    console.log("abc");
    var source = e.target || e.srcElement;
    console.log(source.id);
    var id=source.id;
    var request = $.ajax({
        method: "POST",
        url: "/app/papers/change_lock/",
        data: {id:id},
    });

    request.done(function (data) {
        $("#schedule_div").load(location.href + " #schedule_div", function () {
            $(this).children().unwrap();
            init();
        });
    });
    request.fail(function (XMLHttpRequest, textStatus, errorThrown) {
        alert("Request failed " + textStatus + errorThrown);
        console.log(XMLHttpRequest.responseText)
    });

}

function doSubmit(e){
    var source = e.target || e.srcElement;
    var cname = (source.elements[0].name);
    var cval = (source.elements[0].value);
    var id  = (source.elements[1].value);
    var request = $.ajax({
        method: "POST",
        url: "/app/papers/change_lock/",
        data: {id:id, csrfmiddlewaretoken:cval},
    });
    request.done(function (data) {
        $("#schedule_div").load(location.href + " #schedule_div", function () {
            $(this).children().unwrap();
            init();
            e.preventDefault();
        });
    });
    request.fail(function (XMLHttpRequest, textStatus, errorThrown) {
        alert("Request failed " + textStatus + errorThrown);
        console.log(XMLHttpRequest.responseText)
        e.preventDefault();
    });
    e.preventDefault();
    return request;
};


