function init(){
    data = getData();
}

nv.addGraph(function(){
    var graph = nv.models.scatterChart();
    graph.tooltip.contentGenerator(function(data){
        // data.point contains everything assigned to the point at values.push
        console.log(data.point);
        return data.point.title;
    });
    graph.pointSize(1).pointRange([150,150]);
    //graph.xAxis.tickFormat(d3.format('.0002f'));
    //graph.yAxis.tickFormat(d3.format('.0002f'));

    var data = getData();
    d3.select('svg')
      .datum(data)
      .call(graph);
    nv.utils.windowResize(graph.update);
    return graph;
});

function getData(){
    console.log("aaa");
    coords_x_div = document.getElementById('coordsx').childNodes;
    console.log("bbb");
    coords_y_div = document.getElementById('coordsy').childNodes;
    clusters_div = document.getElementById('clusters').childNodes;
    titles_div = document.getElementById('titles').childNodes;
    coords_x = [];
    coords_y = [];
    clusters = [];
    titles = [];

    for(var i = 0;i < coords_x_div.length; i++){
        if(coords_x_div[i].tagName == "INPUT") {
            coords_x.push(coords_x_div[i].value);
        }
    }
    for(var i = 0;i < coords_y_div.length; i++){
        if(coords_y_div[i].tagName == "INPUT") {
            coords_y.push(coords_y_div[i].value);
        }
    }
    for(var i = 0;i < clusters_div.length; i++){
        if(clusters_div[i].tagName == "INPUT") {
            clusters.push(clusters_div[i].value);
        }
    }
    for(var i = 0;i < titles_div.length; i++){
        if(titles_div[i].tagName == "INPUT") {
            titles.push(titles_div[i].value);
        }
    }

    max_cluster = 0;
    for(var i = 0; i < clusters.length; i++){
        if(clusters[i] > max_cluster){
            max_cluster = clusters[i]
        }
    }
    console.log(coords_x)
    console.log(coords_y)
    var data = []
    var curr_start = 0;
    for(var i = 0;i < max_cluster; i++){
        data.push({
            key: 'Cluster ' + i,
            values: []
        });

        for(var j = 0; j < coords_x.length; j++){
            if(clusters[j] == i+1) {
                console.log("added: " + coords_x[j] + ", " + coords_y[j] + " to " + i)
                data[i].values.push({
                    x: coords_x[j]*100,
                    y: coords_y[j]*100,
                    title: titles[j],
                })
            }
        }
        curr_start += coords_x.length;
    }

    return data;

}

$(document).ready(function () {
    init();
})
