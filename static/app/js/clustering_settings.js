/**
 * Created by Tadej on 6.7.2015.
 */
$(document).ready(function () {
    init();

})

function init() {
    var method = document.getElementById('method');
    method.addEventListener('change', formChange, false);
    $('[name=params]').hide()
    visibleName = method.value
    console.log(visibleName)
    $('#'+visibleName).show()
};

function formChange(){
    console.log(this.value)
    $('[name=params]').hide()
    if (this.value == "kme" || this.value == "kmm" || this.value == "hie"){
        $('#ncl').show()
    } else{
        $('#'+this.value).show()
    }
}