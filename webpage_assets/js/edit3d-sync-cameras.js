/**
 * Syncs camera across <model-viewer> nodes in each .edit3d-example.
 * Only the viewer the user last pressed on ("leader") drives updates, so follower
 * camera-changes do not re-enter sync (avoids freeze / feedback loops).
 *
 * We copy the leader's spherical angles (theta, phi) and orbit radius in *meters*.
 * Followers use that same absolute radius (not a per-asset %).
 *
 * We do **not** copy camera-target to followers: each viewer is its own scene; the
 * orbit pivot must stay that model's center (model-viewer's default). Copying the
 * leader's target into other viewers was orbiting every mesh around the wrong point.
 *
 * Optional on `.edit3d-example`:
 *   data-sync-camera-target="true" — also copy the leader's target (old behavior).
 *
 * Optional per-viewer:
 *   data-sync-orbit-radius — override third component of camera-orbit (`102%` or `1.5m`).
 *   data-camera-target-delta="dx dy dz" — meters added to this viewer's target.
 */
(function () {
  'use strict';

  function readCamera(mv) {
    if (!mv || typeof mv.getCameraOrbit !== 'function') return null;
    try {
      var orbit = mv.getCameraOrbit();
      var target = mv.getCameraTarget();
      return {
        theta: orbit.theta,
        phi: orbit.phi,
        radius: orbit.radius,
        target: target && target.toString ? target.toString() : String(target),
      };
    } catch (err) {
      console.warn('[edit3d-sync] read:', err);
      return null;
    }
  }

  /** model-viewer idle "try dragging" hint — disable for the whole quartet together */
  function clearGroupInteractionPrompts(viewers) {
    for (var i = 0; i < viewers.length; i++) {
      try {
        viewers[i].interactionPrompt = 'none';
      } catch (ignore) {}
    }
  }

  function waitForModelLoad(mv) {
    return new Promise(function (resolve) {
      try {
        if (mv.loaded === true) {
          resolve();
          return;
        }
      } catch (ignore) {}
      var done = function () {
        resolve();
      };
      mv.addEventListener('load', done, { once: true });
      mv.addEventListener('error', done, { once: true });
    });
  }

  function wireGroup(group) {
    var viewers = Array.prototype.slice.call(group.querySelectorAll('model-viewer'));
    if (viewers.length < 2) return;

    function applyCamera(mv, cam) {
      if (!mv || !cam) return;
      try {
        var override = mv.getAttribute('data-sync-orbit-radius');
        var radiusPart =
          override && override.trim()
            ? override.trim()
            : typeof cam.radius === 'number' && !isNaN(cam.radius)
              ? cam.radius + 'm'
              : '105%';
        mv.cameraOrbit =
          cam.theta + 'rad ' + cam.phi + 'rad ' + radiusPart;
        if (group.getAttribute('data-sync-camera-target') === 'true') {
          mv.cameraTarget = cam.target;
        }
        var deltaStr = mv.getAttribute('data-camera-target-delta');
        if (deltaStr) {
          var parts = deltaStr.trim().split(/\s+/);
          var dx = parseFloat(parts[0]);
          var dy = parseFloat(parts[1]);
          var dz = parseFloat(parts[2]);
          if (!isNaN(dx) || !isNaN(dy) || !isNaN(dz)) {
            var t = mv.getCameraTarget && mv.getCameraTarget();
            if (t && typeof t.x === 'number') {
              if (!isNaN(dx)) t.x += dx;
              if (!isNaN(dy)) t.y += dy;
              if (!isNaN(dz)) t.z += dz;
              mv.cameraTarget = t;
            }
          }
        }
      } catch (err) {
        console.warn('[edit3d-sync] apply:', err);
      }
    }

    var leader = null;
    var syncing = false;

    function syncFrom(source) {
      if (syncing) return;
      var cam = readCamera(source);
      if (!cam) return;
      syncing = true;
      try {
        for (var i = 0; i < viewers.length; i++) {
          if (viewers[i] !== source) {
            applyCamera(viewers[i], cam);
          }
        }
        clearGroupInteractionPrompts(viewers);
      } finally {
        syncing = false;
      }
    }

    for (var j = 0; j < viewers.length; j++) {
      (function (mv) {
        mv.addEventListener(
          'pointerdown',
          function () {
            leader = mv;
            clearGroupInteractionPrompts(viewers);
          },
          true
        );

        mv.addEventListener('camera-change', function () {
          try {
            if (leader === null || mv !== leader) {
              return;
            }
            syncFrom(mv);
          } catch (err) {
            console.warn('[edit3d-sync] camera-change:', err);
          }
        });
      })(viewers[j]);
    }

    function alignFollowersToFirst() {
      var cam = readCamera(viewers[0]);
      if (!cam) return;
      syncing = true;
      try {
        for (var k = 1; k < viewers.length; k++) {
          applyCamera(viewers[k], cam);
        }
      } catch (err) {
        console.warn('[edit3d-sync] align:', err);
      } finally {
        syncing = false;
      }
    }

    Promise.all(viewers.map(waitForModelLoad))
      .then(function () {
        try {
          alignFollowersToFirst();
          requestAnimationFrame(function () {
            requestAnimationFrame(alignFollowersToFirst);
          });
        } catch (err) {
          console.warn('[edit3d-sync] initial align:', err);
        }
      })
      .catch(function (err) {
        console.warn('[edit3d-sync] load wait:', err);
      });
  }

  function init() {
    var groups = document.querySelectorAll('.edit3d-example');
    for (var g = 0; g < groups.length; g++) {
      try {
        wireGroup(groups[g]);
      } catch (err) {
        console.warn('[edit3d-sync] wire:', err);
      }
    }
  }

  function run() {
    try {
      if (customElements.get('model-viewer')) {
        init();
      } else {
        customElements.whenDefined('model-viewer').then(init).catch(function (err) {
          console.warn('[edit3d-sync] no model-viewer:', err);
        });
      }
    } catch (err) {
      console.warn('[edit3d-sync] run:', err);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
  } else {
    run();
  }
})();
