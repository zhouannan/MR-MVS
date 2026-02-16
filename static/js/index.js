window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

function showVideoErrorTip(video, message) {
  var existing = video.parentElement.querySelector('.video-error-tip');
  if (existing) return;
  var tip = document.createElement('p');
  tip.className = 'video-error-tip';
  tip.textContent = message;
  video.insertAdjacentElement('afterend', tip);
}

function clearVideoErrorTip(video) {
  var existing = video.parentElement.querySelector('.video-error-tip');
  if (existing) existing.remove();
}

function bindVideoCompatibilityHints() {
  document.querySelectorAll('video').forEach(function(video) {
    var source = video.querySelector('source');
    var src = source ? source.getAttribute('src') : '';
    video.addEventListener('loadedmetadata', function() {
      clearVideoErrorTip(video);
    });
    video.addEventListener('error', function() {
      var err = video.error;
      var reason = '视频加载失败';
      if (err) {
        if (err.code === 2) reason = '网络加载失败';
        if (err.code === 3) reason = '视频解码失败（编码可能不兼容）';
        if (err.code === 4) reason = '浏览器不支持该视频编码';
      }
      showVideoErrorTip(video, reason + '，建议转为 H.264/AAC。文件：' + src);
    });
  });
}

function bindSyncedVideoPair(videoA, videoB) {
  var syncing = false;

  function syncTime(from, to) {
    if (syncing) return;
    if (Math.abs((to.currentTime || 0) - (from.currentTime || 0)) < 0.04) return;
    syncing = true;
    try {
      to.currentTime = from.currentTime;
    } catch (e) {}
    syncing = false;
  }

  function syncPlay(from, to) {
    if (syncing) return;
    syncing = true;
    var p = to.play();
    if (p && typeof p.catch === 'function') p.catch(function() {});
    syncing = false;
  }

  function syncPause(from, to) {
    if (syncing) return;
    syncing = true;
    to.pause();
    syncing = false;
  }

  function syncRate(from, to) {
    if (syncing) return;
    syncing = true;
    to.playbackRate = from.playbackRate;
    syncing = false;
  }

  videoA.addEventListener('play', function() { syncPlay(videoA, videoB); });
  videoB.addEventListener('play', function() { syncPlay(videoB, videoA); });
  videoA.addEventListener('pause', function() { syncPause(videoA, videoB); });
  videoB.addEventListener('pause', function() { syncPause(videoB, videoA); });
  videoA.addEventListener('seeking', function() { syncTime(videoA, videoB); });
  videoB.addEventListener('seeking', function() { syncTime(videoB, videoA); });
  videoA.addEventListener('timeupdate', function() { syncTime(videoA, videoB); });
  videoB.addEventListener('timeupdate', function() { syncTime(videoB, videoA); });
  videoA.addEventListener('ratechange', function() { syncRate(videoA, videoB); });
  videoB.addEventListener('ratechange', function() { syncRate(videoB, videoA); });
}

function bindComparisonVideoSync() {
  document.querySelectorAll('.comparison-group .columns.is-multiline').forEach(function(group) {
    var videos = group.querySelectorAll('video');
    for (var i = 0; i + 1 < videos.length; i += 2) {
      bindSyncedVideoPair(videos[i], videos[i + 1]);
    }
  });
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function updateFourWayClips(container, positions) {
  var p1 = positions[0], p2 = positions[1], p3 = positions[2];
  var layers = container.querySelectorAll('.render-compare-layer');
  function clip(el, leftPct, rightPct) {
    // inset(top right bottom left)
    el.style.clipPath = 'inset(0 ' + rightPct + '% 0 ' + leftPct + '%)';
  }

  layers.forEach(function(layer) {
    var m = layer.getAttribute('data-method');
    if (m === 'transmvsnet') clip(layer, 0, 100 - p1 * 100);
    if (m === 'geomvsnet') clip(layer, p1 * 100, 100 - p2 * 100);
    if (m === 'mvsformerplusplus') clip(layer, p2 * 100, 100 - p3 * 100);
    if (m === 'mrmvs') clip(layer, p3 * 100, 0);
  });

  container.querySelectorAll('.render-compare-handle').forEach(function(h) {
    var idx = parseInt(h.getAttribute('data-handle'), 10) - 1;
    h.style.left = (positions[idx] * 100) + '%';
  });
}

function initFourWayRenderCompare() {
  var container = document.getElementById('render-compare');
  if (!container) return;

  var scanSel = document.getElementById('render-scan-select');
  var frameSel = document.getElementById('render-frame-select');
  var manifestUrl = './static/render_compare/manifest.json';

  var positions = [0.25, 0.5, 0.75];
  updateFourWayClips(container, positions);

  function setAspectFromImage(img) {
    if (!img || !img.naturalWidth || !img.naturalHeight) return;
    container.style.aspectRatio = img.naturalWidth + ' / ' + img.naturalHeight;
  }

  function setImages(scan, frame) {
    container.querySelectorAll('.render-compare-layer').forEach(function(layer) {
      var method = layer.getAttribute('data-method');
      var img = layer.querySelector('img');
      var src = './static/render_compare/' + scan + '/' + method + '/' + frame;
      img.src = src;
      img.onload = function() {
        // Use the first successfully loaded image to set aspect ratio.
        setAspectFromImage(img);
      };
    });
  }

  function refillFrames(frames) {
    frameSel.innerHTML = '';
    frames.forEach(function(f) {
      var opt = document.createElement('option');
      opt.value = f;
      opt.textContent = f;
      frameSel.appendChild(opt);
    });
  }

  function setDefaultSelection(manifest) {
    // Prefer scan1, scan4 if present.
    var scans = Object.keys(manifest.scans || {});
    var preferred = ['scan1', 'scan4'];
    var scan = preferred.find(function(s) { return scans.includes(s); }) || scans[0];
    scanSel.value = scan;
    refillFrames(manifest.scans[scan] || []);
    var frame = frameSel.options.length ? frameSel.options[0].value : '';
    if (frame) setImages(scan, frame);
  }

  function onScanChange(manifest) {
    var scan = scanSel.value;
    refillFrames(manifest.scans[scan] || []);
    var frame = frameSel.options.length ? frameSel.options[0].value : '';
    if (frame) setImages(scan, frame);
  }

  function onFrameChange() {
    var scan = scanSel.value;
    var frame = frameSel.value;
    if (scan && frame) setImages(scan, frame);
  }

  function bindDrag() {
    var activeIdx = null;

    function setFromEvent(e) {
      if (activeIdx == null) return;
      var rect = container.getBoundingClientRect();
      var x = (e.touches && e.touches.length) ? e.touches[0].clientX : e.clientX;
      var t = clamp01((x - rect.left) / rect.width);

      // Keep ordering p1 <= p2 <= p3 with a small gap.
      var gap = 0.02;
      if (activeIdx === 0) t = Math.min(t, positions[1] - gap);
      if (activeIdx === 1) t = Math.max(Math.min(t, positions[2] - gap), positions[0] + gap);
      if (activeIdx === 2) t = Math.max(t, positions[1] + gap);

      positions[activeIdx] = t;
      updateFourWayClips(container, positions);
    }

    function stop() { activeIdx = null; }

    container.querySelectorAll('.render-compare-handle').forEach(function(h) {
      h.addEventListener('mousedown', function(e) {
        activeIdx = parseInt(h.getAttribute('data-handle'), 10) - 1;
        e.preventDefault();
      });
      h.addEventListener('touchstart', function(e) {
        activeIdx = parseInt(h.getAttribute('data-handle'), 10) - 1;
        e.preventDefault();
      }, { passive: false });
    });

    window.addEventListener('mousemove', setFromEvent);
    window.addEventListener('touchmove', setFromEvent, { passive: false });
    window.addEventListener('mouseup', stop);
    window.addEventListener('touchend', stop);
    window.addEventListener('touchcancel', stop);
  }

  fetch(manifestUrl)
    .then(function(r) { return r.json(); })
    .then(function(manifest) {
      scanSel.innerHTML = '';
      Object.keys(manifest.scans || {}).sort().forEach(function(scan) {
        var opt = document.createElement('option');
        opt.value = scan;
        opt.textContent = scan;
        scanSel.appendChild(opt);
      });

      setDefaultSelection(manifest);
      scanSel.addEventListener('change', function() { onScanChange(manifest); });
      frameSel.addEventListener('change', onFrameChange);
      bindDrag();
    })
    .catch(function(err) {
      console.error('Failed to load render manifest:', err);
    });
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();
    bindVideoCompatibilityHints();
    bindComparisonVideoSync();
    initFourWayRenderCompare();

})
