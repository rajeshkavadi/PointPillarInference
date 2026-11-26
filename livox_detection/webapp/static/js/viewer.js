/* PointPillars three.js viewer.
 *
 * Renders a single LiDAR frame (sensor frame, Z up) with:
 *   - the cloud colourised by distance (or height / intensity),
 *   - concentric 5 m radar range rings with distance labels,
 *   - 3D wireframe boxes and per-object class/score/distance labels.
 *
 * Frame layout matches pointpillars/web_server.py `encode_frame`:
 *   magic "LVP2" | u32 points | u32 seq | f64 stamp |
 *   u16 frameIdLen | u16 jsonLen | frameId (padded 4) | json (padded 4) |
 *   points * (f32 x, y, z, intensity)
 */
(function () {
  'use strict';

  var MAGIC = 0x3250564c;          // "LVP2" little endian (single frame)
  var MAGIC_MAP = 0x3350564c;      // "LVP3" little endian (frame + new map points)
  var HEADER_SIZE = 24;
  var HEADER_SIZE_MAP = 28;
  var POINT_BYTES = 16;
  var RING_STEP = 5;
  var RING_COUNT = 14;             // 5 m .. 70 m
  var DEFAULT_CAPACITY = 300000;
  var MAP_INITIAL = 500000;
  var MAP_MAX = 4000000;
  var MAX_BOX_LABELS = 40;

  var RAMP = [
    [0.00, 0.10, 0.24, 0.86],
    [0.25, 0.13, 0.83, 0.93],
    [0.50, 0.13, 0.77, 0.37],
    [0.75, 0.95, 0.72, 0.06],
    [1.00, 0.93, 0.24, 0.24]
  ];

  var CLASS_COLORS = {
    Car: 0x22e2ff,
    Pedestrian: 0xffe23a,
    Cyclist: 0x3dff6a
  };

  var BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
  ];

  var VIEWS = {
    top: { pos: [0, 0.02, 95], target: [0, 0, 0], label: 'TOP · RANGE' },
    iso: { pos: [62, -54, 42], target: [20, 0, 0], label: '3D · PERSPECTIVE' },
    front: { pos: [-24, 0, 12], target: [30, 0, 0], label: 'FRONT · +X' }
  };

  function ramp(t, out, oi) {
    if (t < 0) t = 0;
    else if (t > 1) t = 1;
    var k = 0;
    while (k < RAMP.length - 2 && t > RAMP[k + 1][0]) k++;
    var a = RAMP[k], b = RAMP[k + 1];
    var f = (t - a[0]) / (b[0] - a[0]);
    out[oi] = a[1] + (b[1] - a[1]) * f;
    out[oi + 1] = a[2] + (b[2] - a[2]) * f;
    out[oi + 2] = a[3] + (b[3] - a[3]) * f;
  }

  function rampCss() {
    return 'linear-gradient(90deg,' + RAMP.map(function (s) {
      return 'rgba(' + Math.round(s[1] * 255) + ',' + Math.round(s[2] * 255) +
        ',' + Math.round(s[3] * 255) + ',1) ' + Math.round(s[0] * 100) + '%';
    }).join(',') + ')';
  }

  function View3D(canvas) {
    this.canvas = canvas;
    this.colorMode = 'distance';
    this.maxPoints = 250000;
    this.drawn = 0;
    this.capacity = DEFAULT_CAPACITY;
    this.pose = { x: 0, y: 0 };
    this.poseReady = false;
    this.follow = true;
    this.mapVisible = true;
    this.mapCount = 0;
    this.mapFull = false;
    this._mapEpoch = null;
    this.zMin = -2;
    this.zMax = 6;
    this.dMax = 40;
    this._rangeReady = false;
    this.lastBuffer = null;
    this.onLegend = null;
    this._fps = 0;
    this._frameCount = 0;
    this._fpsFrames = 0;
    this._fpsSince = performance.now();
    this._boxGroup = null;
    this._labels = [];

    this._initScene();
    this._initBuffers(this.capacity);
    this._initMap();
    this._initRings();
    this._bind();
    this.setView('top');
    this._loop();
  }

  /* ------------------------------- scene -------------------------------- */

  View3D.prototype._initScene = function () {
    var width = this.canvas.clientWidth || 800;
    var height = this.canvas.clientHeight || 600;

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance'
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    this.renderer.setClearColor(0x05080d, 1);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, width / height, 0.05, 2500);
    this.camera.up.set(0, 0, 1);

    var grid = new THREE.GridHelper(160, 80, 0x234255, 0x122130);
    grid.rotation.x = Math.PI / 2;
    grid.material.transparent = true;
    grid.material.opacity = 0.55;
    grid.material.depthWrite = false;
    this.grid = grid;
    this.scene.add(grid);

    var origin = new THREE.Mesh(
      new THREE.SphereGeometry(0.14, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xffc451 })
    );
    origin.renderOrder = 6;
    origin.material.depthTest = false;
    this.origin = origin;
    this.scene.add(origin);

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 600;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02;
    this.controls.target.set(0, 0, 0);
  };

  View3D.prototype._initRings = function () {
    this.ringGroup = new THREE.Group();
    this.scene.add(this.ringGroup);

    for (var i = 1; i <= RING_COUNT; i++) {
      var radius = i * RING_STEP;
      var pts = [];
      for (var a = 0; a <= 144; a++) {
        var theta = (a / 144) * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.cos(theta) * radius,
                                   Math.sin(theta) * radius, 0.02));
      }
      var line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({
          color: i % 2 ? 0x3d8fb5 : 0x1e4c62,
          transparent: true,
          opacity: i % 2 ? 0.55 : 0.35,
          depthTest: false,
          depthWrite: false
        })
      );
      line.renderOrder = 3;
      this.ringGroup.add(line);

      // Cardinal spokes make orientation obvious in the top view.
      if (i === RING_COUNT) {
        for (var s = 0; s < 4; s++) {
          var ang = s * Math.PI / 2;
          var spoke = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([
              new THREE.Vector3(0, 0, 0.02),
              new THREE.Vector3(Math.cos(ang) * radius, Math.sin(ang) * radius, 0.02)
            ]),
            new THREE.LineBasicMaterial({
              color: 0x1e4c62, transparent: true, opacity: 0.3,
              depthTest: false, depthWrite: false
            })
          );
          spoke.renderOrder = 3;
          this.ringGroup.add(spoke);
        }
      }

      this.ringGroup.add(this._ringLabel(radius));
    }
  };

  View3D.prototype._ringLabel = function (radius) {
    var canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 64;
    var ctx = canvas.getContext('2d');
    ctx.font = 'bold 38px Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 6;
    ctx.strokeStyle = 'rgba(5, 8, 13, 0.9)';
    ctx.fillStyle = '#7dd8f2';
    ctx.strokeText(radius + ' m', 80, 33);
    ctx.fillText(radius + ' m', 80, 33);

    var texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    var sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: texture, transparent: true, depthTest: false, depthWrite: false
    }));
    var angle = Math.PI / 4;
    sprite.position.set(Math.cos(angle) * radius, Math.sin(angle) * radius, 0.06);
    sprite.scale.set(3.2, 1.28, 1);
    sprite.renderOrder = 4;
    return sprite;
  };

  View3D.prototype._initBuffers = function (capacity) {
    var self = this;
    this.capacity = capacity;
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      this.points.material.dispose();
    }

    this.positions = new Float32Array(capacity * 3);
    this.colors = new Float32Array(capacity * 3);

    var geometry = new THREE.BufferGeometry();
    this.positionAttr = new THREE.BufferAttribute(this.positions, 3);
    this.colorAttr = new THREE.BufferAttribute(this.colors, 3);
    this.positionAttr.setUsage(THREE.DynamicDrawUsage);
    this.colorAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('position', this.positionAttr);
    geometry.setAttribute('color', this.colorAttr);
    geometry.setDrawRange(0, 0);

    var material = new THREE.PointsMaterial({
      size: 0.1,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: false,
      depthWrite: true
    });

    this.points = new THREE.Points(geometry, material);
    this.points.frustumCulled = false;
    this.points.renderOrder = 2;
    this.scene.add(this.points);
    self.drawn = 0;
  };

  /* ------------------------------ map layer ------------------------------ */

  View3D.prototype._initMap = function () {
    this.mapCapacity = MAP_INITIAL;
    this.mapPositions = new Float32Array(this.mapCapacity * 3);
    this.mapColors = new Float32Array(this.mapCapacity * 3);

    var geometry = new THREE.BufferGeometry();
    this.mapPosAttr = new THREE.BufferAttribute(this.mapPositions, 3);
    this.mapColAttr = new THREE.BufferAttribute(this.mapColors, 3);
    this.mapPosAttr.setUsage(THREE.DynamicDrawUsage);
    this.mapColAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('position', this.mapPosAttr);
    geometry.setAttribute('color', this.mapColAttr);
    geometry.setDrawRange(0, 0);

    this.mapCloud = new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 0.09,
      sizeAttenuation: true,
      vertexColors: true,
      depthWrite: true
    }));
    this.mapCloud.frustumCulled = false;
    this.mapCloud.renderOrder = 1;
    this.mapCloud.visible = this.mapVisible;
    this.scene.add(this.mapCloud);
  };

  View3D.prototype._ensureMapCapacity = function (need) {
    if (need <= this.mapCapacity) return;
    var capacity = this.mapCapacity;
    while (capacity < need) capacity = Math.min(capacity * 2, MAP_MAX);
    var pos = new Float32Array(capacity * 3);
    var col = new Float32Array(capacity * 3);
    pos.set(this.mapPositions.subarray(0, this.mapCount * 3));
    col.set(this.mapColors.subarray(0, this.mapCount * 3));
    this.mapPositions = pos;
    this.mapColors = col;
    this.mapCapacity = capacity;
    this.mapPosAttr.array = pos;
    this.mapColAttr.array = col;
    this.mapPosAttr.needsUpdate = true;
    this.mapColAttr.needsUpdate = true;
  };

  /* Appends the new points of one frame; they stay on screen forever. */
  View3D.prototype._appendMap = function (src, count) {
    if (!count || this.mapFull) return 0;
    var room = MAP_MAX - this.mapCount;
    if (count > room) {
      count = room;
      this.mapFull = true;
    }
    if (count <= 0) return 0;
    this._ensureMapCapacity(this.mapCount + count);

    var dst = this.mapPositions, col = this.mapColors;
    var zSpan = this.zMax - this.zMin;
    if (!(zSpan > 0.001)) zSpan = 0.001;
    var invZ = 1 / zSpan;
    var base = this.mapCount * 3;
    for (var i = 0; i < count; i++) {
      var si = i * 4, di = base + i * 3;
      var x = src[si], y = src[si + 1], z = src[si + 2];
      dst[di] = x;
      dst[di + 1] = y;
      dst[di + 2] = z;
      var t = (z - this.zMin) * invZ;
      if (t < 0) t = 0;
      else if (t > 1) t = 1;
      ramp(t * 0.55, col, di);   // dimmer than the live scan
    }
    this.mapCount += count;
    this.mapPosAttr.needsUpdate = true;
    this.mapColAttr.needsUpdate = true;
    this.mapCloud.geometry.setDrawRange(0, this.mapCount);
    return count;
  };

  View3D.prototype._clearMap = function () {
    this.mapCount = 0;
    this.mapFull = false;
    this.mapCloud.geometry.setDrawRange(0, 0);
  };

  View3D.prototype.setShowMap = function (visible) {
    this.mapVisible = !!visible;
    this.mapCloud.visible = this.mapVisible;
  };

  /* -------------------------------- pose --------------------------------- */

  View3D.prototype.setFollow = function (follow) {
    this.follow = !!follow;
    if (this.follow) this._panToPose(this.pose.x, this.pose.y);
  };

  View3D.prototype._panToPose = function (x, y) {
    var dx = x - this.controls.target.x;
    var dy = y - this.controls.target.y;
    this.camera.position.x += dx;
    this.camera.position.y += dy;
    this.controls.target.x = x;
    this.controls.target.y = y;
    this.controls.update();
  };

  View3D.prototype._applyPose = function (pose) {
    if (!pose) return;
    var x = Number(pose.x) || 0;
    var y = Number(pose.y) || 0;
    var previous = this.pose;
    this.pose = { x: x, y: y };

    this.ringGroup.position.set(x, y, 0);
    this.origin.position.set(x, y, 0);
    if (this.grid) {
      this.grid.position.x = Math.round(x / 2) * 2;
      this.grid.position.y = Math.round(y / 2) * 2;
    }
    if (!this.poseReady) {
      this.poseReady = true;
      this.setView(this.viewName || 'top');
      return;
    }
    if (this.follow) {
      this.camera.position.x += x - previous.x;
      this.camera.position.y += y - previous.y;
      this.controls.target.x += x - previous.x;
      this.controls.target.y += y - previous.y;
      this.controls.update();
    }
  };

  /* Put the sensor back in the middle of the view. */
  View3D.prototype.recenter = function () {
    this.setView(this.viewName || 'top');
    this.controls.update();
  };

  /* -------------------------------- input -------------------------------- */

  View3D.prototype._bind = function () {
    var self = this;
    this._onResize = function () { self.resize(); };
    window.addEventListener('resize', this._onResize);
    if (window.ResizeObserver) {
      this._observer = new ResizeObserver(this._onResize);
      this._observer.observe(this.canvas.parentElement || this.canvas);
    }
  };

  View3D.prototype.resize = function () {
    var width = this.canvas.clientWidth || 800;
    var height = this.canvas.clientHeight || 600;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  };

  /* ------------------------------- options ------------------------------- */

  View3D.prototype.setColorMode = function (mode) {
    this.colorMode = mode;
    this._rangeReady = false;
    if (this.lastBuffer) this.applyFrame(this.lastBuffer, true);
  };

  View3D.prototype.setMaxPoints = function (limit) {
    this.maxPoints = limit;
    if (this.lastBuffer) this.applyFrame(this.lastBuffer, true);
  };

  View3D.prototype.setShowRings = function (visible) {
    this.ringGroup.visible = !!visible;
  };

  View3D.prototype.setShowLabels = function (visible) {
    this.labelsVisible = !!visible;
    for (var i = 0; i < this._labels.length; i++) {
      this._labels[i].visible = this.labelsVisible;
    }
  };

  View3D.prototype.setView = function (name) {
    var preset = VIEWS[name] || VIEWS.top;
    var px = this.pose.x, py = this.pose.y;
    this.camera.position.set(px + preset.pos[0], py + preset.pos[1], preset.pos[2]);
    this.controls.target.set(px + preset.target[0], py + preset.target[1],
                             preset.target[2]);
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(this.controls.target);
    this.controls.update();
    this.viewName = name;
    this.onView && this.onView(preset.label, name);
  };

  /* -------------------------------- frame -------------------------------- */

  View3D.prototype.applyFrame = function (buffer, isRecolor) {
    if (!buffer || buffer.byteLength < HEADER_SIZE) return false;
    var view = new DataView(buffer);
    var magic = view.getUint32(0, true);
    var isMap = magic === MAGIC_MAP;
    if (!isMap && magic !== MAGIC) return false;
    var headerSize = isMap ? HEADER_SIZE_MAP : HEADER_SIZE;

    var total = view.getUint32(4, true);
    var seq = view.getUint32(8, true);
    var stamp = view.getFloat64(12, true);
    var fidLen = view.getUint16(20, true);
    var jsonLen = view.getUint16(22, true);
    var mapTotal = isMap ? view.getUint32(24, true) : 0;
    var fidOff = headerSize;
    var jsonOff = fidOff + ((fidLen + 3) & ~3);
    var dataOff = jsonOff + ((jsonLen + 3) & ~3);
    var mapOff = dataOff + total * POINT_BYTES;
    if (!total || mapOff + mapTotal * POINT_BYTES > buffer.byteLength) return false;

    var doc = { boxes: [], meta: {}, frame_id: '' };
    try {
      if (jsonLen > 0 && window.TextDecoder) {
        doc = JSON.parse(new TextDecoder('utf-8').decode(
          new Uint8Array(buffer, jsonOff, jsonLen)));
      }
    } catch (err) { /* keep an empty annotation set */ }

    var src = new Float32Array(buffer, dataOff, total * 4);
    var stride = this.maxPoints > 0 ? Math.ceil(total / this.maxPoints) : 1;
    var count = Math.ceil(total / stride);

    if (count > this.capacity) {
      this._initBuffers(Math.max(count, this.capacity * 2));
    }

    // Pass 1: range of this frame (percentile based, so outliers don't wash
    // out the colour ramp).
    var lo = Infinity, hi = -Infinity, far = 0;
    var hist = new Uint32Array(101);
    var valid = 0;
    for (var s = 0; s < total; s += stride) {
      var si = s * 4;
      var z = src[si + 2];
      if (!isFinite(z)) continue;
      if (z < lo) lo = z;
      if (z > hi) hi = z;
      var d = Math.sqrt(src[si] * src[si] + src[si + 1] * src[si + 1] + z * z);
      if (!isFinite(d)) continue;
      valid++;
      hist[d >= 100 ? 100 : (d | 0)]++;
    }
    if (!isFinite(lo) || !isFinite(hi)) return false;
    if (valid) {
      var target = valid * 0.99, acc = 0;
      for (var b = 0; b < hist.length; b++) {
        acc += hist[b];
        if (acc >= target) { far = b + 1; break; }
      }
    }
    var smooth = this._rangeReady && !isRecolor ? 0.15 : 1;
    this.zMin += (lo - this.zMin) * smooth;
    this.zMax += (hi - this.zMax) * smooth;
    this.dMax += (Math.max(far, 20) - this.dMax) * smooth;
    this._rangeReady = true;
    if (this.zMax - this.zMin < 0.4) {
      var mid = (this.zMax + this.zMin) / 2;
      this.zMin = mid - 0.2;
      this.zMax = mid + 0.2;
    }

    // Map mode: the scan is in map coordinates, so move the sensor markers and
    // append the points that are new to the map.  Not on a recolour pass: the
    // same buffer must not be added twice.
    if (isMap && !isRecolor) {
      var meta = doc.meta || {};
      var epoch = meta.map_epoch;
      if (epoch !== undefined && epoch !== null) {
        if (this._mapEpoch !== null && epoch !== this._mapEpoch) this._clearMap();
        this._mapEpoch = epoch;
      } else if (meta.map_reset) {
        this._clearMap();
      }
      this._applyPose(doc.pose);
      if (mapTotal) {
        this._appendMap(new Float32Array(buffer, mapOff, mapTotal * 4), mapTotal);
      }
    }

    // Pass 2: positions + colours.
    var dst = this.positions, col = this.colors;
    var mode = this.colorMode;
    var zSpan = this.zMax - this.zMin;
    if (!(zSpan > 0.001)) zSpan = 0.001;
    var invZ = 1 / zSpan;
    var invD = 1 / (this.dMax > 1 ? this.dMax : 1);

    for (var i = 0; i < count; i++) {
      var srcI = (i * stride) * 4;
      var dstI = i * 3;
      var x = src[srcI], y = src[srcI + 1], zz = src[srcI + 2];
      dst[dstI] = x;
      dst[dstI + 1] = y;
      dst[dstI + 2] = zz;

      var t;
      if (mode === 'distance') {
        t = Math.sqrt(x * x + y * y + zz * zz) * invD;
      } else if (mode === 'height') {
        t = (zz - this.zMin) * invZ;
      } else {
        t = src[srcI + 3] / 255;
      }
      ramp(t, col, dstI);
    }

    this.positionAttr.updateRange.offset = 0;
    this.positionAttr.updateRange.count = count * 3;
    this.positionAttr.needsUpdate = true;
    this.colorAttr.updateRange.offset = 0;
    this.colorAttr.updateRange.count = count * 3;
    this.colorAttr.needsUpdate = true;
    this.points.geometry.setDrawRange(0, count);

    this.drawn = count;
    this.lastBuffer = buffer;
    this._frameSeq = seq;
    this._stamp = stamp;
    this._frameId = doc.frame_id || '';
    this._meta = doc.meta || {};
    this._frameCount++;
    this._fpsFrames++;
    this._updateBoxes(doc.boxes || []);
    this._emitStats();
    this._emitLegend();
    return true;
  };

  View3D.prototype._emitLegend = function () {
    if (!this.onLegend) return;
    if (this.colorMode === 'height') {
      this.onLegend('HEIGHT (Z)', this.zMin.toFixed(1) + ' m',
        this.zMax.toFixed(1) + ' m', ((this.zMin + this.zMax) / 2).toFixed(1) + ' m');
    } else if (this.colorMode === 'distance') {
      this.onLegend('RANGE', '0 m', this.dMax.toFixed(0) + ' m',
        (this.dMax / 2).toFixed(0) + ' m');
    } else {
      this.onLegend('INTENSITY', '0', '255', '128');
    }
  };

  View3D.prototype._emitStats = function () {
    if (!this.onStats) return;
    var now = performance.now();
    if (now - this._fpsSince >= 1000) {
      this._fps = this._fpsFrames * 1000 / (now - this._fpsSince);
      this._fpsSince = now;
      this._fpsFrames = 0;
    }
    this.onStats({
      fps: this._fps,
      points: this.drawn,
      boxes: this._boxCount || 0,
      mapPoints: this.mapCount,
      mapFull: this.mapFull,
      pose: this.pose,
      frameId: this._frameId,
      meta: this._meta || {},
      seq: this._frameSeq
    });
  };

  /* -------------------------------- boxes -------------------------------- */

  View3D.prototype._updateBoxes = function (boxes) {
    if (this._boxGroup) {
      this.scene.remove(this._boxGroup);
      this._boxGroup.traverse(function (child) {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (child.material.map) child.material.map.dispose();
          child.material.dispose();
        }
      });
      this._boxGroup = null;
    }
    this._labels = [];
    this._boxCount = boxes.length;

    var group = new THREE.Group();
    if (!boxes.length) {
      this._boxGroup = group;
      this.scene.add(group);
      return;
    }

    var pos = new Float32Array(boxes.length * BOX_EDGES.length * 2 * 3);
    var col = new Float32Array(boxes.length * BOX_EDGES.length * 2 * 3);
    var p = 0, c = 0;

    for (var i = 0; i < boxes.length; i++) {
      var box = boxes[i];
      var corners = this._corners(box);
      var color = new THREE.Color(CLASS_COLORS[box.c] || 0xffffff);
      for (var e = 0; e < BOX_EDGES.length; e++) {
        var a = corners[BOX_EDGES[e][0]];
        var b = corners[BOX_EDGES[e][1]];
        pos[p++] = a[0]; pos[p++] = a[1]; pos[p++] = a[2];
        pos[p++] = b[0]; pos[p++] = b[1]; pos[p++] = b[2];
        for (var k = 0; k < 2; k++) {
          col[c++] = color.r; col[c++] = color.g; col[c++] = color.b;
        }
      }

      if (i < MAX_BOX_LABELS) {
        var dist = Math.sqrt(box.x * box.x + box.y * box.y + box.z * box.z);
        var label = this._boxLabel(box, dist, color);
        label.visible = this.labelsVisible !== false;
        group.add(label);
        this._labels.push(label);
      }
    }

    var geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(col, 3));
    var lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.95, depthTest: true
    }));
    lines.renderOrder = 5;
    group.add(lines);

    this._boxGroup = group;
    this.scene.add(group);
  };

  View3D.prototype._corners = function (box) {
    var hx = box.dx / 2, hy = box.dy / 2, hz = box.dz / 2;
    var cos = Math.cos(box.yaw), sin = Math.sin(box.yaw);
    var local = [
      [hx, hy, hz], [-hx, hy, hz], [-hx, -hy, hz], [hx, -hy, hz],
      [hx, hy, -hz], [-hx, hy, -hz], [-hx, -hy, -hz], [hx, -hy, -hz]
    ];
    var out = [];
    for (var i = 0; i < 8; i++) {
      var lx = local[i][0], ly = local[i][1], lz = local[i][2];
      out.push([
        box.x + lx * cos - ly * sin,
        box.y + lx * sin + ly * cos,
        box.z + lz
      ]);
    }
    return out;
  };

  View3D.prototype._boxLabel = function (box, dist, color) {
    var canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 128;
    var ctx = canvas.getContext('2d');
    var text = box.c + ' ' + Number(box.s).toFixed(2) + '  ·  ' +
      dist.toFixed(1) + ' m';
    ctx.font = 'bold 46px Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 8;
    ctx.strokeStyle = 'rgba(5, 8, 13, 0.9)';
    ctx.strokeText(text, 256, 68);
    ctx.fillStyle = '#' + color.getHexString();
    ctx.fillText(text, 256, 68);

    var texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    var sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: texture, transparent: true, depthTest: false, depthWrite: false
    }));
    sprite.position.set(box.x, box.y, box.z + box.dz / 2 + 0.9);
    sprite.scale.set(9, 2.25, 1);
    sprite.renderOrder = 7;
    return sprite;
  };

  /* ------------------------------ main loop ------------------------------ */

  View3D.prototype._loop = function () {
    var self = this;
    function frame() {
      self.controls.update();
      self.renderer.render(self.scene, self.camera);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  };

  /* ============================ page wiring ============================== */

  var viewer = null;
  var els = {};
  var reconnectTimer = null;

  function $(id) { return document.getElementById(id); }

  function setWsState(online, text) {
    var pill = $('pill-ws');
    var dot = $('hud-dot');
    if (pill) pill.classList.toggle('on', online);
    if (dot) dot.classList.toggle('on', online);
    $('hud-ws').textContent = text || (online ? 'live' : 'offline');
  }

  function boot() {
    els = {
      overlay: $('overlay'),
      legendTitle: $('legend-title'),
      legendMin: $('legend-min'),
      legendMax: $('legend-max'),
      legendMid: $('legend-mid'),
      viewTitle: $('view-title'),
      viewSub: $('view-sub')
    };
    $('legend-bar').style.background = rampCss();

    viewer = new View3D($('view'));
    viewer.labelsVisible = true;

    viewer.onLegend = function (title, min, max, mid) {
      els.legendTitle.textContent = title;
      els.legendMin.textContent = min;
      els.legendMax.textContent = max;
      els.legendMid.textContent = mid;
    };
    viewer.onView = function (label) {
      els.viewTitle.textContent = label;
    };
    viewer.onStats = function (stats) {
      $('hud-points').textContent = stats.points.toLocaleString();
      $('hud-boxes').textContent = stats.boxes;
      $('pill-points').textContent = stats.points.toLocaleString();
      $('pill-boxes').textContent = stats.boxes;
      $('pill-frame').textContent = stats.frameId || ('#' + stats.seq);
      $('hud-fps').textContent = stats.fps.toFixed(1);
      $('hud-map').textContent = (stats.mapPoints || 0).toLocaleString();
      $('pill-map').textContent = (stats.mapPoints || 0).toLocaleString();
      var infer = stats.meta && stats.meta.infer_ms;
      if (infer) $('pill-infer').textContent = Math.round(infer);
    };

    $('color-mode').addEventListener('change', function (ev) {
      viewer.setColorMode(ev.target.value);
    });
    $('max-points').addEventListener('change', function (ev) {
      viewer.setMaxPoints(parseInt(ev.target.value, 10) || 0);
    });
    $('show-rings').addEventListener('change', function (ev) {
      viewer.setShowRings(ev.target.checked);
    });
    $('show-labels').addEventListener('change', function (ev) {
      viewer.setShowLabels(ev.target.checked);
    });
    $('show-map').addEventListener('change', function (ev) {
      viewer.setShowMap(ev.target.checked);
    });
    $('follow').addEventListener('change', function (ev) {
      viewer.setFollow(ev.target.checked);
    });
    $('recenter').addEventListener('click', function () {
      viewer.recenter();
    });

    var buttons = document.querySelectorAll('.btn[data-view]');
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].addEventListener('click', function (ev) {
        for (var j = 0; j < buttons.length; j++) {
          buttons[j].classList.remove('active');
        }
        ev.target.classList.add('active');
        viewer.setView(ev.target.getAttribute('data-view'));
      });
    }

    connect();
    pollStatus();
  }

  function connect() {
    if (reconnectTimer) return;
    var scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
    var ws;
    try {
      ws = new WebSocket(scheme + location.host + '/ws');
    } catch (err) {
      reconnectTimer = setTimeout(function () { reconnectTimer = null; connect(); }, 1500);
      return;
    }
    ws.binaryType = 'arraybuffer';
    ws.onopen = function () {
      setWsState(true, 'live');
    };
    ws.onmessage = function (ev) {
      if (viewer.applyFrame(ev.data) && els.overlay) {
        els.overlay.classList.add('hidden');
      }
    };
    ws.onclose = function () {
      setWsState(false, 'offline');
      reconnectTimer = setTimeout(function () { reconnectTimer = null; connect(); }, 1500);
    };
    ws.onerror = function () {
      try { ws.close(); } catch (err) { /* ignore */ }
    };
  }

  function pollStatus() {
    fetch('/api/status', { cache: 'no-store' })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (!data) return;
        if (data.age_s === null || data.age_s > 10) {
          setWsState(data.clients > 0, data.clients > 0 ? 'waiting' : 'offline');
        } else {
          setWsState(true, 'live');
        }
        var meta = data.meta || {};
        var pose = meta.pose;
        els.viewSub.textContent = (data.source === 'ros' ? 'ROS topics' : 'bag replay') +
          (pose ? ' · sensor ' + Number(pose.x).toFixed(1) + ', ' +
            Number(pose.y).toFixed(1) + ' m' : '') +
          ' · map ' + (meta.map_points || 0).toLocaleString() + ' pts';
      })
      .catch(function () { /* server briefly unavailable */ })
      .then(function () { setTimeout(pollStatus, 2000); });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
