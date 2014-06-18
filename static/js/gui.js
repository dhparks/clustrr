// script which runs the interactive elements of the page
// expects d3 and jquery ($)

var gui = {},

    // colors
    flickrBlue = '#006add',
    flickrPink = '#ff1981',
    
    initialRaster = 9,
    
    // analysis progress vars
    getUser,
    scrapeData,
    buildNetwork,
    eigenValues,
    trainSOM,
    assignments,
    centralities,
    download,
    sequence,
    sequenceCount = 0;

// this is the master object. it holds all the interactive objects
// once they are created, as well as data received from the server
gui.data = {};
gui.locks = {'canDrag':true, 'canClick':true, 'canChangeTags':true};
gui.data.selectedClusters = [];
gui.data.selectedTags = [];
gui.data.photosForTags = {};
gui.data.selectedPhotoIds = {};
gui.data.selectedPhotoThumbs = [];
gui.data.selectedPhotoLinks = [];
gui.data.oldCount = 0;

// ==============================
// ANALYSIS PROGRESS SECTION: functions which manage
// the progress of the analysis on the server. analysisStage
// manages the sending and receiving of data; send and success
// functions are specified for each stage
// ==============================

/** @constructor */
function analysisStage(label, caption, send, success) {
    
    // analysisStage class only has one method
    this.execute = function () {
        
        var i, t;
        
        function _callback(data) {
            if (data.status === 'ok') {
                i.css("background-color",flickrBlue);
                t.success(data);
                nextStage();
            }
            if (data.status === 'fail') {
                i.css("background-color",flickrPink);
            }
        }
        
        // set the div color to pending
        i = $("#"+this.label);
        i.css("background-color","orange");

        // scoping!
        t = this;
        
        // sending the query changes the state of data on the server,
        // so use post.
        $.post('clustrr/'+this.label, this.send(), _callback);
    };

    // generic defaults
    if (success === undefined) {
        this.success = genericSuccess;
        } else {
            this.success = success;
    }
    
    if (send === undefined) {
        this.send = genericSend;
        } else {
            this.send = send;
    }
    
    this.label = label;
    this.caption = caption;
}

function genericSuccess(data) {}

function userSuccess(data) {
    gui.data.nsid = data.nsid;
}

function downloadSuccess(data) {
    
    // executed when data is successfully returned from the server
    
    var t = ['members', 'tags', 'centralities', 'counts', 'tagsToPhotos',
             'photosToThumbs','photosToLinks'];
    
    function makeEigenPlot() {
        // eigenplot
        if (gui.eigenplot === undefined) {
            gui.eigenplot = new eigenplot('eigenplot', 295, 245);
            gui.eigenplot.create();
        }
        
        gui.eigenplot.data = data.eigenvalues.map(function (d, i) {return {x:i+1, y:parseFloat(d)};});
        gui.eigenplot.setScales();
        gui.eigenplot.redraw();
        gui.eigenplot.replot();
    }
    
    function makeSOM() {
        // som (data must be passed at creation

        if (gui.som !== undefined) {
            delete gui.som;
            d3.select("#som-svg").remove();
            $("#som").off('click'); // without this there are duplicate listeners
        }

        gui.som = new som('som', 245, 245, data.umatrix, data.borders, data.clusters);
        gui.som.create();
        gui.som.outline(initialRaster);
        gui.som.assignClusters(initialRaster);
    }
    
    function makeBlocks() {
        // block diagonal
        if (gui.blocks === undefined) {
            gui.blocks = new clusterGraph('clustergraph', 245, 245);
            gui.blocks.create();
        }
        
        gui.blocks.clusterData = data.blockdiagonal;
        gui.blocks.frameSizeX = data.ntags;
        gui.blocks.frameSizeY = data.ntags;
        gui.blocks.loadImage(data.blockdiagonalurl,initialRaster);
        gui.blocks.raster(initialRaster);
    }
    
    function makeWordCloud() {
        //word cloud
        if (typeof gui.wordcloud === "undefined") {
            gui.wordcloud = new wordCloud('wordcloud', 780, 200);
            gui.wordcloud.create();
        }
        
        gui.wordcloud.clear();
        d3.select("#wordcloud").style("opacity",1);
        
        // photocounter (lives inside wordcloud)
        if (gui.photocounter === undefined) {
            gui.photocounter = new photoCounter('wordcloud');
            gui.photocounter.create();
        }
        
    }
    
    function makeMatrix() {
        //photo matrix for showing tagged photos
        if (gui.photoMatrix === undefined) {
            gui.photoMatrix = new photoMatrix('photoMatrix', 3, 5);
            gui.photoMatrix.create();
        }
        
        d3.selectAll('.thumbnail_element').remove();
    }
    
    function make() {
        makeEigenPlot();
        makeSOM();
        makeBlocks();
        makeWordCloud();
        makeMatrix();
    }
    
    // once we've received the data, we can create
    // the analytical displays and attach their data.
    $.map(t, function (d,i) {gui.data[d] = data[d];});
    
    // retract the progress bar using jquery effect
    $("#progress2").slideUp(400, make);
}

function genericSend() {
    return {};
}

function userSend() {
    var u = $("#username");
    return {'name':u.val() || u.attr("placeholder")};
}

function nextStage() {
    if (sequenceCount < sequence.length) {
        sequence[sequenceCount].execute();
        sequenceCount = sequenceCount+1;
    }
}

// define each command to the server using an instance of analysisStage
getUser = new analysisStage('getuser', ['Find', 'User'], userSend, userSuccess);
scrapeData = new analysisStage('scrape', ['Scrape', 'Data']);
buildNetwork = new analysisStage('buildnetwork', ['Build', 'Network']);
eigenValues = new analysisStage('eigenvalues', ['Calculate', 'Spectrum']);
trainSOM = new analysisStage('trainsom', ['Train', 'SOMs']);
assignments = new analysisStage('assignments', ['Assign', 'Clusters']);
centralities = new analysisStage('centralities', ['Calculate', 'Centrality']);
download = new analysisStage('download', ['Download', 'Data'], genericSend, downloadSuccess);

// the sequence to run analysis commands
sequence = [getUser,scrapeData,buildNetwork,eigenValues,trainSOM,assignments,centralities,download];

// ==============================
// INTERACTIVE GUI OBJECTS SECTION
// ==============================

// object for the word cloud of the selected cluster(s)
/** @constructor */
function wordCloud(where, width, height) {
    // object attributes follow
    this.where = where;
    this.width = width;
    this.height = height;
    this.cachedVocabulary = {};
    
    // color interpolators for the centrality measurement
    this.color1 = d3.rgb(flickrBlue);
    this.color2 = d3.rgb(flickrPink);
    this.interpolator = d3.interpolateRgb(this.color1, this.color2);
    
    // object methods follow
    this.create = function () {
        // create the svg element in the DOM
        d3.select('#'+this.where).append("svg")
            .attr("id", this.where+'-svg')
            .attr("width", this.width)
            .attr("height", this.height);
            
        d3.select('#'+this.where+'-svg')
            .append("g")
            .attr("transform", "translate("+0+","+0+")")
            .attr("id", "sandbox")
            .style("opacity", 0);
    };
    
    this.clear = function () {
        // clear the wordcloud box. used for redrawing.
        //d3.selectAll('#'+this.where+'-svgg text').remove()
        d3.selectAll('#sandbox text').remove();
    };
    
    this.clusterSelectDeselect = function(cluster) {
        // if cluster is not in cachedVocabulary, generate
        // the vocabulary datastructure for it and add it
        // to the cache. then use the cache to generate
        // a vocabulary to draw on the wordcloud using drawToFit.

        var newVocab = [], // new word objects added when a cluster is selected
            assembledVocab = [], // 
            k, // loop variable
            word, // tmp in building vocab
            s, // tmp in building vocab
            fill; // color of words
        
        if (!this.cachedVocabulary.hasOwnProperty(cluster) &&
            gui.data.members.hasOwnProperty(cluster)) {
            // vocab structure: [{word:word, size:count, fill:centrality}, {}...]
            for (k=0; k<gui.data.members[cluster].length; k++) {
                word = gui.data.tags[gui.data.members[cluster][k]];
                fill = this.interpolator(gui.data.centralities[cluster][word]/255);
                newVocab.push({'word':word, 'size':gui.data.counts[word], 'fill':fill});
            }
            this.cachedVocabulary[cluster] = newVocab;
        }
        
        // build the composite vocabulary from all the selected clusters
        for (k=0; k<gui.data.selectedClusters.length; k++) {
            s = this.cachedVocabulary[gui.data.selectedClusters[k]];
            assembledVocab = assembledVocab.concat(s);
        }
        
        //draw the cloud
        this.drawToFit(assembledVocab);
    };
    
    this.drawToFit = function(vocabulary) {
        // draw several word clouds until a good scaling of the font
        // size is achieved.

        d3.select('#sandbox').style("opacity",0);

        var sandbox = d3.select("#sandbox"), // where to draw
            fits = true, // tracks when we make the size too big
            scaler = 2, // how much to change the guess by
            scale = 1, // initial guess for scaling
            tune = 5, // how many loops in finetuning
            k, // loop variable for finetuning
            upperBound, // for finetuning
            lowerBound, // for finetuning
            count = 0, // loop counter, course tuning
            tries = 0; // times to try drawing before restarting

        if (vocabulary.length > 0) {
            
            this.clear();

            // exponentially increase the size of the scaling factor until
            // not all the words can fit
            while (fits) {
                this.redraw(vocabulary, scale);
                if (d3.selectAll("#sandbox text")[0].length < vocabulary.length) {
                    fits = false;
                } else {
                    scale *= scaler;
                }
                this.clear();
                count = count + 1;
            }
            
            // if the letters are too big to start, we need more iterations
            // in the tuning stage
            if (count === 1) {
                tune = 8;
                upperBound = 1;
                lowerBound = 0;
            } else {
                tune = 5;
                upperBound = scale;
                lowerBound = scale/scaler;
            }

            // perform a binary division search to optimize the size
            for (k = 0; k < tune; k++) {
                scale = (upperBound+lowerBound)/2;
                this.redraw(vocabulary, scale);
                if (d3.selectAll("#sandbox text")[0].length < vocabulary.length) {
                    upperBound = scale;
                } else {
                    lowerBound = scale;
                }
                this.clear();
            }
            
            // redraw at a size known to be ok
            while (d3.selectAll("#sandbox text")[0].length < vocabulary.length) {
                this.clear();
                this.redraw(vocabulary,lowerBound);
                if (tries > 10) {
                    // start over
                    this.drawToFit(vocabulary);
                }
                tries = tries+1;
            }

            this.center();
            d3.select("#sandbox").style("opacity",1);

            // recolor the selected tags
            for (k = 0; k < gui.data.selectedTags.length; k++){
                d3.select("#wordcloud-"+gui.data.selectedTags[k])
                    .style("fill","orange")
                    .classed("selected",true);
            }
        } else {
            gui.locks.canClick = true;
        }
        
        gui.photocounter.update();
    };
    
    this.center = function() {
        // centers the wordcloud
        
        var sbBB = d3.select("#sandbox")[0][0].getBBox(), // bounding box of sandbox
            wc = d3.select("#wordcloud-svg")[0][0], // bounding box of wordcloud
            x = -sbBB.x+(wc.offsetWidth-sbBB.width)/2, // x translate
            y = -sbBB.y+(wc.offsetHeight-sbBB.height)/2; // y translate
        
        // execute the translation transform
        d3.select("#sandbox").attr("transform","translate("+x+","+y+")");
    };
    
    this.redraw = function(vocabulary, scale) {

        function draw(words) {
            
            d3.select('#sandbox')
                .selectAll("text")
                .data(words)
                .enter()
                .append("text")
                    .style("font-size", function(d) {return d.size + "px"; })
                    .style("font-family", "Impact")
                    .style("fill", function(d, i) {return d.fill;})
                    .attr("text-anchor", "middle")
                    .attr("transform", function(d) {return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";})
                    .attr("id",function(d) {return "wordcloud-"+d.text;})
                    .attr("data-fill",function(d) {return d.fill;})
                    .text(function(d) {return d.text;})
                    .on("click",function(d) {selectTag(d.text);});
                    
            gui.locks.canClick = true;
        }

        gui.locks.canClick = false;
        
        d3.layout.cloud().size([this.width, this.height])
            .words(vocabulary.map(function(d) {return {text: d.word, fill: d.fill, size: d.size*scale};}))
            .rotate(0)
            .font("Impact")
            .fontSize(function(d) { return d.size; })
            .on("end",draw)
            .start();
    };

}

// object for displaying how many photos are in the intersection
// of the selected tags. lives in wordCloud.
/** @constructor */
function photoCounter(where) {
    // class attributes follow
    this.where = where;
    this.parent = d3.select('#'+this.where+'-svg');
    this.oldCount = 0;
    
    // class methods follow
    this.create = function () {
        // create some dummy text; see how big it is; change
        // font size to fill height
        var h1, // height of 12px text
            h2, // height of bounding box
            s; // scale factor
        
        this.parent
            .append("text")
                .attr("id", "counter-text")
                .style("font-size", "12px")
                .style("font-family", "Impact")
                .style("fill", "black")
                .style("opacity", 1)
                .attr("x", "50%")
                .attr("y", "50%")
                .style("opacity", 0)
                .text("0")
                .attr("text-anchor", "middle")
                .attr("alignment-baseline", "central");
                
        // rescale vertically by changing the font size
        h1 = d3.select("#counter-text")[0][0].getBBox().height;
        h2 = this.parent[0][0].offsetHeight;
        s = 12*h2/h1;
        d3.select("#counter-text").style("font-size", s+"px").style("opacity", 0.3);
    };
        
    this.update = function () {
        // change the text of the number counter using
        // d3 tween/text interpolator
    
        var count, // number of photos
            t = this, // for scoping
            keys, // selected photo ids
            k, // loop variable
            i, // interpolator between old and new counts
            p = [], // tmp for photo thumbs
            l = []; // tmp for photo links

        // update the text
        count = Object.keys(gui.data.selectedPhotoIds).length || 0;

        d3.select("#counter-text")
            .transition()
            .duration(300)
            .tween("text", function () {
                var i = d3.interpolateRound(t.oldCount, count);
                return function (t) {
                    this.textContent = i(t);
                };
            })
            .each("end", function () {t.oldCount = count;});

        keys = Object.keys(gui.data.selectedPhotoIds);
        for (k = 0; k < count; k++) {
            l.push(gui.data.photosToLinks[keys[k]]);
            p.push(gui.data.photosToThumbs[keys[k]]);
        }
        
        gui.data.selectedPhotoThumbs = p;
        gui.data.selectedPhotoLinks = l;
    };
}

// object for displaying the u-matrix represenation of
// the self-organizing map and the borders between clusters.
/** @constructor */
function som(where, width, height, pixelData, borderData, clusterData) {
    // **** object attributes follow **** //
    this.where = where;
    this.width = width;
    this.height = height;
    this.borderColor = flickrPink;
    
    // pixel data should be a 2d array of pixel intensities
    this.pixelData = pixelData;
    
    // these variables allow caching of the strings
    // used with d3 to change the border colors, improving
    // redraw speed by about 30%. border data should be
    // a 3d array where the first index is the number of clusters-1
    this.borderData = borderData;
    this.borderStrings = {};
    
    // clusters data should be a 3d array where the first index
    // is the number of clusters-1. value of any entry is the
    // cluster assignment.
    this.clusterData = clusterData;

    // **** object methods follow ***** //
    this.create = function () {
        // creates the SOM svg
        d3.select('#'+this.where)
            .append("svg")
            .attr("id", this.where+'-svg')
            .attr("width", this.width)
            .attr("height", this.height);
            
        d3.select("#"+this.where+'-svg')
            .append("g")
            .attr("id", this.where+'-rectGroup');
            
        // precalculate the border information for
        // speedy scrubbing
        this._precalculateOutlines();
        
        // draw the som
        this.draw(this.pixelData);
    };
      
    // helper function for precalculateOutlines
    this._buildOutlineString = function (data, label) {

        var tops    = '', // track cells with top border
            rights  = '', // track cells with right border
            bottoms = '', // track cells with bottom border
            lefts   = '', // track cells with left border
            topCodes = [4,5,6,7,12,13,14,15], // codes indicating a top border
            rightCodes = [1,3,5,7,9,11,13,15], // codes indicating a right border
            bottomCodes = [8,9,10,11,12,13,14,15], // codes indicating a bottom border
            leftCodes = [2,3,6,7,10,11,14,15], // codes indicating a left border
            i, j, // loop variables
            v, d; // string building
            
        for (i = 0; i < data.length; i++) {
            for (j = 0; j < data[i].length; j++) {
                v = data[i][j];
                if (v > 0) {
                    d = ',#som_row'+i+'_col'+j;
                    if (topCodes.indexOf(v) > -1) {
                        tops = tops+d;
                    }
                    if (rightCodes.indexOf(v) > -1) {
                        rights = rights+d;
                    }
                    if (bottomCodes.indexOf(v) > -1) {
                        bottoms = bottoms+d;
                    }
                    if (leftCodes.indexOf(v) > -1) {
                        lefts = lefts+d;
                    }
                }
            }
        }
    
        this.borderStrings[label] = {
            'tops':tops.slice(1),
            'rights':rights.slice(1),
            'bottoms':bottoms.slice(1),
            'lefts':lefts.slice(1)};
    };
    
    // precalculate outline strings used by d3 to
    // change cluster outline status
    this._precalculateOutlines = function () {
        var k; // loop var
        for (k = 0; k < this.borderData.length; k++) {
            this._buildOutlineString(this.borderData[k],k.toString());
        }
        // don't need this anymore; free up some memory
        delete this.borderData;
    };
    
    // change the opacity of the border elements to indicate
    // cluster assignments
    this.outline = function(borderIndex) {
        // borderNumber is the key to borderStrings which
        // stores the precalculated strings used by d3 to
        // alter the outlining rect elements. should be
        // an integer or a string
        
        // further performance could be achieved in this
        // operation if we instead had pre-calculated list of
        // CHANGES when transitioning from one set of clusters
        // to another.
        
        var s; // border string

        // select all the borders in the SOM and deselect
        d3.selectAll("#"+this.where).selectAll(".border").style("fill-opacity", 0);
        
        // select borders by side and style
        s = this.borderStrings[borderIndex.toString()];
        d3.selectAll(s.tops).selectAll(".top").style("fill-opacity", 1);
        d3.selectAll(s.rights).selectAll(".right").style("fill-opacity", 1);
        d3.selectAll(s.bottoms).selectAll(".bottom").style("fill-opacity", 1);
        d3.selectAll(s.lefts).selectAll(".left").style("fill-opacity", 1);
    };
    
    // draw DOM/SVG elements
    this.draw = function (cellData) {
        // data should be a 2d list of intensities
        // assume an equal number of columns in each row
        this.scaleY = this.height/cellData.length;
        this.scaleX = this.width/cellData[0].length;
        
        var x1 = this.scaleX-1,
            y1 = this.scaleY-1,
            x2 = this.scaleX-2,
            y2 = this.scaleY-2,
            rectList = [],
            bc = this.borderColor,
            v, k, l, cs, borders;

        // build strings specifying cell data (positions, colors, ids)
        for (k = 0; k < cellData.length; k++) {
            for (l = 0; l < cellData[k].length; l++) {
                v = 255-cellData[k][l];
                rectList.push({'t':'translate('+this.scaleX*l+','+this.scaleY*k+')',
                              'id':'som_row'+k+'_col'+l,
                              'v':v,
                              'b':'rgb('+v+','+v+','+v+')'});
            }
        }

        // remove old rects
        d3.select('#'+this.where+'-rectGroup').selectAll('g').remove();

        // draw new rects. each cell in the SOM is a group of 9
        // rectangles: the big body pixel and the border pixels
        d3.select('#'+this.where+'-rectGroup').selectAll('g')
            .data(rectList)
            .enter()
            .append('g')
            .attr('id', function (d) {return d.id;})
            .attr('transform', function (d) {return d.t;})
            .attr("data-selected", false)
            .attr("data-v", function (d) {return d.v;})
            .classed("somcell", true);

        cs = d3.selectAll(".somcell");
        
        // this is the main body rectangle
        cs.append('rect')
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", this.scaleX)
            .attr("height", this.scaleY)
            .attr("fill", function (d) {return d.b;})
            .attr("fill-opacity", 1)
            .classed("center", true);

        // these are the border rectangles; to outline a border,
        // we select by id and class and change fill opacity.
        // for example, this colors top borders of two cells:
        // d3.selectAll("#rowX_colY, #rowA_colB").selectAll(".top").style("fill-opacity",1)
        borders = [
                    {'x':1, 'y':0, 'w':x2,'h':1, 'c':"border top"},
                    {'x':x1,'y':1, 'w':1, 'h':y2,'c':"border right"},
                    {'x':1, 'y':y1,'w':x2,'h':1, 'c':"border bottom"},
                    {'x':0, 'y':1, 'w':1, 'h':y2,'c':"border left"},
                    {'x':0, 'y':0, 'w':1, 'h':1, 'c':"border top left"},
                    {'x':x1,'y':0, 'w':1, 'h':1, 'c':"border top right"},
                    {'x':x1,'y':y1,'w':1, 'h':1, 'c':"border bottom right"},
                    {'x':0, 'y':y1,'w':1, 'h':1, 'c':"border bottom left"}];

        $.map(borders, function(d,i) {
            cs.append('rect')
                .attr("x", d.x)
                .attr("y", d.y)
                .attr("width", d.w)
                .attr("height", d.h)
                .style("fill", bc)
                .style("fill-opacity", 0)
                .classed(d.c, true);
        });
        
        // attach a click action to the groups
        $('#'+this.where).on('click','.somcell', this.clickCallback);
        
    };
    
    // assign values to data-clusters attribute fields of som groups
    this.assignClusters = function(clustersIndex) {

        d3.selectAll(".somcell").attr("data-cluster",null);

        var cdata = this.clusterData[clustersIndex],
            cells = [],
            d,v,i,j;
        
        for (i = 0; i < cdata.length; i++){
            for (j = 0; j < cdata[i].length; j++){
                d = '#som_row'+i+'_col'+j;
                v = cdata[i][j];
                cells.push([d,v]);
            }
        }
        $.map(cells, function (v,i) {$(v[0]).attr('data-cluster',v[1]);});
    };
    
    this.clusterSelectDeselect = function (cluster) {
        
        var p,
            ds,
            color,
            selected,
            _coloring = function (t,d) {
                // recolors selected elements; also marks a change in
                // the data-selected attribute of the parent
                p  = d3.select(t.parentNode);
                ds = p.attr("data-selected");
                if (ds === 'false') {
                    color = d3.rgb('hsl(211, 60%,'+100.0*d.v/255*0.8+'%)');
                    selected = 'true';
                } else {
                    color = "rgb("+d.v+','+d.v+','+d.v+")";
                    selected = 'false';
                }
                p.attr("data-selected",selected);
                return color;
            };
    
        // if clusters have been assigned, select every pixel
        // that is in the same cluster as this one. recolor them.
        d3.selectAll("[data-cluster='"+cluster+"']")
            .select(".center")
            .attr("fill", function (d) {return _coloring(this, d);});
    };

    // define the behavior that results from clicking on a pixel    
    this.clickCallback = function() {
        //selectCluster($(this).data('cluster')) <-- this breaks!
        selectCluster(parseInt(d3.select(this).attr("data-cluster")));
    };
}

// object for the block-diagonal cluster plot. adapted
// from the rasterBackground object in speckle_server guiObjects
/** @constructor */
function clusterGraph(where, width, height) {
    
    // **** object attributes **** //
    
    this.name = where;
    this.width = width;
    this.height = height;
    this.rasterValue = initialRaster; // default view: 10 clusters
    
    // size of each frame in pixels; needs to be
    // filled in by gui when the data is downloaded.
    this.frameSizeX = 0;
    this.frameSizeY = 0;
    
    // gets set externally when data comes in
    this.clusterData = null;
    
    // **** object methods **** //
    this.create = function () {
        // create svg element with an img and another
        // svg which holds the clickable tiles
        d3.select('#'+this.name)
            .append("svg")
            .attr("id", this.name+'-svg')
            .attr("width", this.width)
            .attr("height", this.height);
            
        d3.select('#'+this.name+'-svg')
            .append("image")
            .attr("id", this.name+'-img');
            
        d3.select("#"+this.name+'-svg')
            .append("g")
            .attr("id", this.name+'-rectGroup');
    };
    
    this.loadImage = function (path, rasterValue) {

        // load image into background. only gets called
        // once, when data is received from server. later,
        // raster gets called to move the image around
    
        function setAttr(t) {
    
            // calculate the correct scaling factors
            t.scaleX = t.width/t.frameSizeX;
            t.scaleY = t.height/t.frameSizeY;
            t.scaleStr = 'scale('+t.scaleX+','+t.scaleY+')';
        
            var w = t.image.width,
                h = t.image.height;
            t.gridSize = w/t.frameSizeX;

            // set all the image attributes
            d3.select('#'+t.name+'-img')
                .attr("width", w)
                .attr("height", h)
                .attr("xlink:href", t.image.src)
                .attr('transform', t.scaleStr);
            }
            
        function onload(t, rasterValue) {
            setAttr(t);
            t.raster(rasterValue);
            t.drawOverlays(rasterValue);
        }

        // load a background off the server and set it to the
        // correct image attribute
        var t = this;
        if (rasterValue === undefined) {
            rasterValue=this.rasterValue;
        }
        this.image = new Image();
        this.image.onload = function () {onload(t, rasterValue);};
        this.image.src = path;
    };
    
    this.raster = function (n) {
        // move around the background image
        // n is the frame number found in some other action
        
        var ix, // x-index of translations
            iy, // y-index of translation
            str; // translation string
        
        ix = n%this.gridSize;
        iy = Math.floor(n/this.gridSize);
        str = this.scaleStr+' translate(-'+this.frameSizeX*ix+',-'+this.frameSizeY*iy+')';
        
        // apply transform
        $("#"+this.name+'-img').attr('transform',str);
        this.rasterValue = n;
    };
    
    this.drawOverlays = function (n) {
        
        // there should be as many sets of row borders as
        // there are tuples. to keep computational cost
        // low, this function should only be called at the
        // end of the rastering operation, as a user cannot
        // interact with the scrubber and the cluster selector
        // at the same time
        
        // rowData should have the following format:
        // [(cluster number, number of rows)]
        // cluster number identifies the cluster so that
        // we can look up tags. it is important that the cluster be
        // in the correct position in the list.
        
        // jquery cant handle SVG so we first build a list of information
        // arrays for each rectangle, then let d3 do the drawing.
        // array information: [width, height, x, y, clusterX, clusterY]
        
        // styling is determined through the clusterOverlay class

        if (this.clusterData !== null) {
            
            // declare variables
            var data = this.clusterData[n],
                cumulative = 0,
                cumulatives = [],
                clusters = [],
                sizes = [],
                rectList = [],
                k, i, j, x;

            // cumulative sum to correctly position rects
            for (k = 0; k < data.length; k++) {
                clusters.push(data[k][0]);
                sizes.push(data[k][1]);
                cumulatives.push(cumulative);
                cumulative += data[k][1];
            }
            
            // rect attributes; parsed by d3 in drawing
            for (i = 0; i < data.length; i++) {
                for (j = 0; j < data.length; j++) {
                    x = {'width':this.scaleX*sizes[j],
                        'height':this.scaleY*sizes[i],
                        'x':this.scaleX*cumulatives[j],
                        'y':this.scaleY*cumulatives[i],
                        'c1':clusters[i],
                        'c2':clusters[j],
                        'classes':'clusterOverlay CGunselected'+' cluster'+clusters[i]+' cluster'+clusters[j]};
                    if (i === j) {
                        x.classes = x.classes+' diagonalCluster';
                    }
                    rectList.push(x);
                }
            }        
            
            // remove old rects
            d3.select("#"+this.name+"-rectGroup").selectAll('rect').remove();
            
            // draw new rects
            d3.select('#'+this.name+'-rectGroup').selectAll('rect')
                .data(rectList)
                .enter()
                .append('rect')
                .attr("width", function(d) {return d.width;})
                .attr("height", function(d) {return d.height;})
                .attr("x", function(d) {return d.x;})
                .attr("y", function(d) {return d.y;})
                .attr('data-clusters', function(d) {return [d.c1,d.c2];})
                .attr("class", function(d) {return d.classes;});
        }
        
    };
    
    this.clusterSelectDeselect = function (n) {
        var j = '[data-clusters="'+n+','+n+'"]', // selector for this cluster
            t = d3.selectAll(j), // select
            x = t.classed("CGunselected"); // are they in the class?
    
        // toggle classes
        t.classed("CGunselected", !x);
        t.classed("CGselected", x);
    };

    this.clickCallback = function () {
        // we only select/deselect if the click occurs
        // on the diagonal
        var d = $(this).attr("data-clusters").split(',');
        if (d[0] === d[1]) {
            selectCluster(parseInt(d[0]));
        }   
    };
    
    // this attaches the callback for when we click on elements
    $('#'+this.name).on('click', 'rect', this.clickCallback);
    
}

// graph object. for eigenvalues
/** @constructor */
function eigenplot(where, width, height) {
    // **** object atttributes **** //

    this.where = where;
    this.width = width;
    this.height = height;
    
    this.margins = {'top': 18, 'right': 10, 'bottom': 5, 'left': 24};

    this.strokeColor = "black";
    this.strokeWidth = 1.5;

    this.data = null; // needs to be filled in
    
    this.setScales = function () {
        // call this whenever data gets set; creates/replaces
        // the scaling functios so data is drawn correctly

        var x = this.data.map(function (d) {return d.x;}),
            y = this.data.map(function (d) {return d.y;});
        
        this.domainMin = Math.min.apply(null, x);
        this.domainMax = Math.max.apply(null, x);
        this.rangeMin = Math.min.apply(null, y);
        this.rangeMax = Math.max.apply(null, y);
        
        this.xScale = d3.scale.linear()
                        .range([this.margins.left, this.width-this.margins.right])
                        .domain([this.domainMin, this.domainMax]);
                        
        this.yScale = d3.scale.linear()
                        .range([this.height-this.margins.top, this.margins.bottom])
                        .domain([this.rangeMin, this.rangeMax]);
                        
        this.lineFunc = d3.svg.line().interpolate("linear")
                        .x(function(d) {return this.xScale(d.x); })
                        .y(function(d) {return this.yScale(d.y); });
    };
        
    this.create = function () {
        // populate the div with the correct svg
        d3.select("#"+this.where).append("svg")
            .attr("width", this.width)
            .attr("height", this.height)
            .attr("id", this.where+"-svg");
    };
    
    this.resetSVG = function () {
        // first, delete the old plot
        d3.select("#"+this.where+"-group").remove();
        
        // now, re-add the svg group
        d3.select("#"+this.where+"-svg")
            .append("g")
            .attr("id", this.where+"-group");
    };
    
    this.drawGrids = function (where) {
        
        //draw grid lines

        var svga = d3.select("#"+this.where+"-group"),
            t = this;

        svga.append("g").attr("id", where+"-verticalGrid");
        
        d3.select("#"+this.where+"-verticalGrid").selectAll(".gridlines")
            .data(t.xScale.ticks())
            .enter()
            .append("line")
            .attr("class", "gridlines")
            .attr("x1", function (d) {return t.xScale(d);})
            .attr("x2", function (d) {return t.xScale(d);})
            .attr("y1", function ()  {return t.yScale(t.rangeMin);})
            .attr("y2", function ()  {return t.yScale(t.rangeMax);});

        svga.append("g").attr("id",this.where+"-horizontalGrid");
        
        d3.select("#"+this.where+"-horizontalGrid").selectAll(".gridlines")
            .data(t.yScale.ticks(5)).enter()
            .append("line")
            .attr("class", "gridlines")
            .attr("x1", function () {return t.xScale(t.domainMin);})
            .attr("x2", function () {return t.xScale(t.domainMax);})
            .attr("y1", function (d) {return t.yScale(d);})
            .attr("y2", function (d) {return t.yScale(d);});
    };
    
    this.drawAxes = function (where, extras) {
        
        // the extras argument is how we distinguish the axes behavior for different plot
        // types. optionally, this could be overridden by the child but this seems
        // difficult within the prototype inheritance scheme

        var svga, xAxis, yAxis, xText, yText, xa_ty, ya_tx;
        
        svga = d3.select("#"+this.where+"-group");
        xa_ty = this.height-this.margins.top;
        ya_tx = this.margins.left;
        
        if (extras === undefined) {
            extras = {};
        }
        
        // nticksX
        if ('nticksX' in extras) {
            xAxis = d3.svg.axis().scale(this.xScale).orient("bottom").ticks(extras.nticksX);
        } else {
            xAxis = d3.svg.axis().scale(this.xScale).orient("bottom").ticks(5);
        }
        
        // nticksY
        if ('nticksY' in extras) {
            yAxis = d3.svg.axis().scale(this.yScale).orient("left").ticks(extras.nticksY);
        } else {
            yAxis = d3.svg.axis().scale(this.yScale).orient("left").ticks(5);
        }
            
        // yText
        if ('yText' in extras) {
            yText = extras.yText;
        } else {
            yText = 'Scaled eig. val.';
        }
        
        // xText
        if ('xText' in extras) {
            xText = extras.xText;
        } else {
            xText = 'Eig. val. #';
        }
        
        // draw x axis
        svga.append("g")
            .attr("class", "x plotaxis")
            .attr("transform", "translate(0,"+xa_ty+")")
            .call(xAxis);

        // draw y axis
        svga.append("g")
            .attr("class", "y plotaxis")
            .attr("transform", "translate("+ya_tx+",0)")
            .call(yAxis);

        // draw xaxis text
        svga.append("text")
            .attr("y", this.height-this.margins.top-5)
            .attr("x", this.width-this.margins.right-4)
            .attr("font-size",12)
            .style("text-anchor", "end")
            .text(xText);
        
        // draw yaxis text
        svga.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", this.margins.right+28)
            .attr("x", -5)
            .attr("dy", ".05em")
            .attr("font-size", "12")
            .style("text-anchor", "end")
            .text(yText);
    };

    this.redraw = function (extras) {
        // break these down into smallers parts so that they
        // can be overridden easily by children graphs
        this.resetSVG(this.where);
        this.drawGrids(this.where);
        this.drawAxes(this.where, extras);
    };
    
    this._sliderStartAction = function () {
        deselectAllClusters();
    };
    
    // dragging action for the slider
    this._sliderDragAction = function () {
        
        var b,
            oldx,
            newx,
            x2;
        
        // get the old position
        b = d3.select("#"+this.where+'-slider');
        oldx = parseFloat(b.attr("data-x"));
        newx = oldx+d3.event.dx;
        
        // update the svg elements for eigenplot
        if (newx < this.margins.left) {
            newx = this.margins.left;
        }
        
        if (newx > this.width-this.margins.right) {
            newx = this.width-this.margins.right;
        }
        
        b.attr("data-x",newx);
        b.attr("transform","translate("+newx+',0)');
        
        // raster the background of clustergraph
        x2 = Math.floor(this.xScale.invert(newx))-1;
        
        gui.blocks.raster(x2);
        
        // re-outline the SOM
        gui.som.outline(x2);
    };
    
    // dragend action for the slider
    this._sliderEndAction = function () {
        // when the slider stops moving, we want to
        // 1. redraw the cluster overlays on the blocks diagram
        // 2. snap the selector to the nearest integer
        
        var s, // slider selected
            x1, // x values
            x2,
            x3,
            tween; // tweening function for transition
        
        s = d3.select("#"+this.where+'-slider');
        
        // calculate the x value of the nearest lesser point
        x1 = parseFloat(s.attr("data-x")); 
        x2 = Math.floor(this.xScale.invert(x1));
        x3 = this.xScale(x2);
        
        // transition
        if (x2 > 0) {
            tween = function (d, i, a) {
                return d3.interpolateString(a, "translate(" + x3 + ",0)");
            };
            s.transition().duration(100).delay(0).attrTween("transform", tween);
            
            // send the redraw command to blocks
            gui.blocks.drawOverlays(x2-1);
            
            // reassign the clusters on the som
            gui.som.assignClusters(x2-1);
        }
    };

    this.replot = function (where, args) {
    
        var t = this, // for scoping
            db, // drag behavior
            xvals, // x points
            yvals, // y points
            svga; // svg for the plot
    
        function _getColor() {
            var color;
            try {
                color = args.color;
            }
            catch (err) {
                color = t.strokeColor;
            }
            return color;
        }
        
        function _getWidth() {
            try {
                width = args.width;
            }
            catch (err) {
                width = t.strokeWidth;
            }
            return width;
        }

        svga = d3.select('#'+this.where+'-group');
    
        // clear the old plot
        svga.select("#"+this.where+'-plot').remove();
        svga.append("g").attr("id", this.where+"-plot");
        
        // make the new plot
        d3.select("#"+t.where+"-plot").selectAll(".selecter")
            .data([t.data])
            .enter()
            .append("path")
            .attr("d", function (d) {return t.lineFunc(d);})
            .attr("fill", "none")
            .attr("stroke", _getColor())
            .attr("stroke-width", _getWidth());
            
        // add circles
        d3.select("#"+t.where+"-plot").selectAll("circle")
            .data(t.data)
            .enter()
            .append("circle")
            .attr("cx", function (d) {return t.xScale(d.x);})
            .attr("cy", function (d) {return t.yScale(d.y);})
            .attr("r", 2)
            .attr("fill", flickrPink)
            .attr("stroke", _getColor())
            .attr("stroke-width", 1);
        
        // dragging behavior for slider
        db = d3.behavior.drag()
            .origin(function () {
                var x = d3.select(this);
                return {x: x.attr("data-x"), y:0};
                })
            .on("dragstart", function () {
                if (gui.locks.canDrag) {
                    t._sliderStartAction();}
                })
            .on("drag", function () {
                if (gui.locks.canDrag) {
                    t._sliderDragAction(this.id);}
                })
            .on("dragend", function () {
                if (gui.locks.canDrag) {
                    t._sliderEndAction(this.id);}
            });
            
        // draw the slider in its own group
        svga.append('g')
            .attr("id", this.where+'-slider')
            .attr("data-x", this.xScale(initialRaster+1))
            .attr("transform", "translate("+this.xScale(initialRaster+1)+",0)")
            .call(db);
            
        xvals = [0];
        d3.select("#"+this.where+"-slider").selectAll(".line")
            .data(xvals)
            .enter()
            .append("line")
            .classed("slider", true)
            .attr("x1", function (d) {return d;})
            .attr("x2", function (d) {return d;})
            .attr("y1", function () {return t.yScale(t.rangeMin);})
            .attr("y2", function () {return t.yScale(t.rangeMax);})
            .attr("stroke", flickrBlue)
            .attr("stroke-width", 2);
            
        yvals = [t.rangeMin,t.rangeMax];
        d3.select("#"+this.where+"-slider").selectAll("circle")
            .data(yvals)
            .enter()
            .append("circle")
            .classed("slider", true)
            .attr("cx", 0)
            .attr("cy", function (d) {return t.yScale(d);})
            .attr("r", 4.5)
            .attr("fill", flickrBlue);
            
    };
}

// object for displaying thumbnails
/** @constructor */
function photoMatrix(where, rows, cols) {
    
    this.where = where;
    this.rows = rows;
    this.cols = cols;
    this.urls = []; // this is where we store the urls for photos. array.
    this.dummy = "static/images/placeholder.png";
    this.currentStart = 0;
    
    this.create = function () {
        
        // create the row divs
        var str = '',
            r,
            str2,
            t=this;
            
        for (r = 0; r < this.rows; r++) {
            str = str+'<div class="thumbnail_row" id="thumbnail_row_'+r+'"></div>';
        }
        
        // scroll arrows
        str2 = '<div id="scrollarrows">' +
                '<img id="scrolldown" class="scrollinactive" src="static/images/downarrow.png">' +
                '<img id="scrollup" class="scrollinactive" src="static/images/uparrow.png"></div>';
        $("#"+this.where).append(str+str2);
        
        // attach actions to the imgs
        $('#scrolldown').click(function () {t.scroll('down');});
        $('#scrollup').click(function () {t.scroll('up');});
    };
    
    this.scroll = function (direction) {
        
        if (direction === 'down') {
            if (this.urls.length-this.currentStart > this.rows*this.cols) {
                this.redraw(this.currentStart+this.rows*this.cols);
            }
        }
        
        if (direction === 'up') {
            if (this.currentStart >= this.rows*this.cols) {
                this.redraw(this.currentStart-this.rows*this.cols);
            }
        }
    };

    this.redraw = function (start) {
        
        var kmax, // how many photos to display in this round
            rmax, // how many rows of photos
            r, // row counter
            x, // temp var for DOM manipulation
            k, // photo counter
            rstrs = [], // row strings to add to DOM
            fmt = '<div class="thumbnail_element">' +
                    '<a href="URL" target="_blank">' +
                    '<img src="SOURCE" width="150" height="150">' +
                    '</a></div>'
                    .replace('USER',gui.data.nsid);
        
        // remove all the images
        d3.selectAll(".thumbnail_element").remove();
        
        // get the current record of images
        this.urls = gui.data.selectedPhotoThumbs;
        this.links = gui.data.selectedPhotoLinks;

        kmax = Math.min(this.urls.length-start,this.rows*this.cols);
        rmax = Math.ceil(kmax/this.cols);
        
        // build row strings
        for (r = 0; r < rmax; r++) {
            rstrs.push('');
        }
        
        k = 0;
        while (k < kmax) {
            r = Math.floor(k/this.cols);
            rstrs[r] += fmt.replace(/SOURCE/g,this.urls[k+start]).replace(/URL/g,this.links[k+start]);
            k += 1;
        }
        
        // add the elements and fix the width. don't know how to do this properly with css
        for (r = 0; r < rmax; r++) {
            x = $('#thumbnail_row_'+r);
            x.append();
            $('#thumbnail_row_'+r).append(rstrs[r]);
            $('#thumbnail_row_'+r).css("width",x[0].children.length*152).css("height",152);
        }
        
        for (r = rmax; r < this.rows; r++) {
            $('#thumbnail_row_'+r).css("height",0);
        }
        
        // record our current start point
        this.currentStart = start;
        
        // activate buttons if applicable
        if (this.currentStart >= this.rows*this.cols) {
            d3.select("#scrollup").classed("scrollactive",true);
            d3.select("#scrollup").classed("scrollinactive",false);
        } else {
            d3.select("#scrollup").classed("scrollactive",false);
            d3.select("#scrollup").classed("scrollinactive",true);
        }
        
        if (this.urls.length-this.currentStart > this.rows*this.cols) {
            d3.select("#scrolldown").classed("scrollactive",true);
            d3.select("#scrolldown").classed("scrollinactive",false);
        } else {
            d3.select("#scrolldown").classed("scrollactive",false);
            d3.select("#scrolldown").classed("scrollinactive",true);
        }
    };
}

// ======================================================
// USER INTERACTIONS requiring mediation between objects)
// ======================================================

function selectCluster(cluster) {

    if (gui.locks.canClick) {
        // set a lock to avoid a race condition being created
        // by the expensive wordcloud drawing operation
        
        var t = [],
            i,
            k = gui.data.selectedClusters.indexOf(cluster);

        // build list of tags
        cluster = parseInt(cluster);
        for (i = 0; i < gui.data.members[cluster].length; i++) {
            t.push(gui.data.tags[gui.data.members[cluster][i]]);
        }
        
        if (k > -1) {
            // deselecting a cluster
            gui.data.selectedClusters.splice(k,1);
            for (k = 0; k < t.length; k++) {
                if (gui.data.selectedTags.indexOf(t[k]) > -1) {
                    selectTag(t[k],true);
                }
            }

            gui.data.selectedPhotoIds = intersection(gui.data.selectedTags,gui.data.tagsToPhotos) || {};
            gui.photocounter.update();
            gui.photoMatrix.redraw(0);
            
        } else {
            gui.data.selectedClusters.push(cluster);
        }
        
        // given a cluster number, dispatch select/deselect commands to each
        // of the interactive elements
        gui.som.clusterSelectDeselect(cluster);
        gui.blocks.clusterSelectDeselect(cluster);
        gui.wordcloud.clusterSelectDeselect(cluster);
    }
}

function selectTag(tag, defer) {
    if (gui.locks.canChangeTags) {
        
        var k,
            i,
            y;
        
        if (defer === undefined) {
            defer = false;
        }

        // track the selected tags
        k = gui.data.selectedTags.indexOf(tag);
        if (k > -1) {
            gui.data.selectedTags.splice(k, 1);
        } else {
            gui.data.selectedTags.push(tag);
        }

        // change the color and class of the text
        y = d3.select('#wordcloud-'+tag);
        if (k > -1) {
            y.style("fill", y.attr("data-fill"));
        } else {
            y.style("fill", "orange");
        }
        y.classed("selected", !y.classed("selected"));
        
        // calculate photo id intersection
        if (!defer) {
            gui.data.selectedPhotoIds = intersection(gui.data.selectedTags, gui.data.tagsToPhotos) || {};
            gui.photocounter.update();
            gui.photoMatrix.redraw(0);
        }
    } 
}

function deselectAllClusters() {
    // this gets called by the slider action
    
    var i; // loop var
    gui.locks.canDrag = false;
    
    for (i = 0; i < gui.data.selectedClusters.length; i++) {
        gui.som.clusterSelectDeselect(gui.data.selectedClusters[i]);
        gui.blocks.clusterSelectDeselect(gui.data.selectedClusters[i]);
    }
    
    // remove old rects
    d3.select('#'+gui.blocks.name+'-rectGroup').selectAll('rect').remove();
    
    gui.data.selectedClusters = [];
    gui.data.selectedTags = [];
    gui.data.selectedPhotoIds = {};
    gui.wordcloud.clusterSelectDeselect();
    gui.photoMatrix.redraw(0);
    gui.locks.canDrag = true;
    
}

// =====================================================
// OTHER STUFF
// =====================================================

function buildTable () {
    // build the html for the progress table, then append it
    // with jquery to the dom
    var tr1 = '',
        tr2 = '',
        tr3 = '',
        f = '',
        base1 = '<td colspan="3">CAPTION</td>',
        base2 = '<td class="crhr"></td><td><div class="indicator" id="LABEL"></div></td><td class="crhr"></td>',
        r1,r2,r3,i;
    
    for (i = 0; i < sequence.length; i++) {
        tr1 = tr1+base1.replace(/CAPTION/g,sequence[i].caption[0]);
        tr2 = tr2+base2.replace(/LABEL/g,sequence[i].label);
        tr3 = tr3+base1.replace(/CAPTION/g,sequence[i].caption[1]);
    }
    
    r1 = '<tr class="textrow">'+tr1+'</tr>';
    r2 = '<tr class="circlerow">'+tr2+'</tr>';
    r3 = '<tr class="textrow">'+tr3+'</tr>';
    f  = '<table>'+r1+r2+r3+'</table>';
    
    $('#progress2').append(f);
    $(".crhr").css("width", 100);//(815-12*sequence.length)/(2*sequence.length))
}

function intersection(keys, source) {
    // tags is a list of tags. we want
    // to calculate all the photos that the
    // tags have in common through intersect.
    
    var k, // loop variables
        x, // tmp for helper
        r = source[keys[0]]; // initial list of keys
    
    function helper(o1, o2) {
        // get common keys of two objects
        var i;
        x = {};
        for (i in o1) {
            if (o2.hasOwnProperty(i)) {
                x[i] = null;
            }
        }
        return x;
    }

    for (k = 1; k < keys.length; k++) {
        r = helper(r, source[keys[k]]);
    }
    
    return r;
}
    
function union(tags) {
    // return the union of all the photos associated
    // with the tags in tags
    
    var r = {}, // list of keys
        k, // loop variable
        o; // object key
    
    for (k = 0; k < tags.length; k++) {
        for (o in gui.data.tagsToPhotos[tags[k]]) {
            r[o] = 1;
        }
    }
    
    return r;
}

function resetAndGo() {
	
	// reset data
	gui.data = {};
	gui.locks = {'canDrag':true, 'canClick':true, 'canChangeTags':true};
	gui.data.selectedClusters = [];
	gui.data.selectedTags = [];
	gui.data.photosForTags = {};
	gui.data.selectedPhotoIds = {};
	sequenceCount = 0;
	nextStage();
    }

// ====================================================
// RUN THE INTERFACE: build the progress table
// and attach click action which starts the analysis
// ====================================================
$(document).ready( function () {
    
    buildTable();

    $('#flickruser').submit(function () {
        // start analysis process
        event.preventDefault();
        $(".indicator").css("background-color", "inherit");
        $("#progress2").slideDown(200, resetAndGo);
    
    });
});




