function transpose(a) {

    // Calculate the width and height of the Array
    var w = a.length || 0;
    var h = a[0] instanceof Array ? a[0].length : 0;

    // In case it is a zero matrix, no transpose routine needed.
    if (h === 0 || w === 0) { return []; }

    /**
     * @var {Number} i Counter
     * @var {Number} j Counter
     * @var {Array} t Transposed data is stored in this array.
     */
    var i, j, t = [];

    // Loop through every item in the outer array (height)
    for (i = 0; i < h; i++) {

        // Insert a new row (array)
        t[i] = [];

        // Loop through every item per item in outer array (width)
        for (j = 0; j < w; j++) {

            // Save transposed data.
            t[i][j] = a[j][i];
        }
    }

    return t;
}

function disp_chart(chartdata) {
    var allRows = chartdata.split(/\r?\n|\r/);
    for (var singleRow = 0; singleRow < allRows.length; singleRow++) {
        allRows[singleRow] = allRows[singleRow].split(',');
    }
    allRows = allRows.splice(1);
    allRows = transpose(allRows);
    mydata = {
        labels: allRows[0],
        datasets: [{
            data: allRows[3],
            label: "confidence",
            borderColor: "#c45850",
            fill: false
        }, {
            data: allRows[11],
            label: "gaze_angle_x",
            borderColor: "#3e95cd",
            fill: false
        }, {
            data: allRows[12],
            label: "gaze_angle_y",
            borderColor: "#8e5ea2",
            fill: false
        }, {
            data: allRows[297],
            label: "pose_Rx",
            borderColor: "#3cba9f",
            fill: false
        }, {
            data: allRows[298],
            label: "pose_Ry",
            borderColor: "#e8c3b9",
            fill: false
        }]
    };
    new Chart(document.getElementById('myChart').getContext('2d'), {
        type: 'line',
        data: mydata,
        options: {
            animation: {
                duration: 0
            }
        }
    });
}