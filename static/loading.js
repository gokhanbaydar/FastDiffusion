$(document).ready(function () {
    let loadingModalTimerId;
    let loadingModalStartTime;
    function updateLoadingModalTimer(currentTime) {
        const elapsedTime = (currentTime - loadingModalStartTime) / 1000;
        document.getElementById('loading_modal_timer').textContent = "elapsed " + elapsedTime.toFixed(1) + " seconds";
        document.getElementById('loading_modal_elapsed_seconds').textContent = "elapsed " + elapsedTime.toFixed(1) + " seconds";
        loadingModalTimerId = requestAnimationFrame(updateLoadingModalTimer);
    }
    function starteLoadingModalTimer() {
        loadingModalStartTime = performance.now();
        requestAnimationFrame(updateLoadingModalTimer);
    }
    $('#loading_modal').on('show.bs.modal', function (event) {
        starteLoadingModalTimer();
    });
    $('#loading_modal').on('hide.bs.modal', function (event) {
        cancelAnimationFrame(loadingModalTimerId);
    });
});

function showLoadingModal() {
    $(document).ready(function () {
        $('#loading_modal').modal('show');
    });
}

function hideLoadingModal() {
    $('#loading_modal').modal('hide');
}

function showErrorModal(responseError) {
    $(document).ready(function () {
        $('#error_modal').modal('show');
        $("#error_modal_text").html(responseError.responseText);
    });
}