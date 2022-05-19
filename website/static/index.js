function deleteNote(noteId) {
  fetch("/delete-note", {
    method: "POST",
    body: JSON.stringify({ noteId: noteId }),
  }).then((_res) => {
    window.location.href = "/";
  });
}
function test() {
  var notifyloading = "Waiting for uploading..."
  Notiflix.Loading.init({
    className: "notiflix-loading",
    zindex: 4000,
    backgroundColor: "rgba(0,0,0,0.8)",
    rtl: false,
    fontFamily: "Inherit",
    cssAnimation: true,
    cssAnimationDuration: 400,
    clickToClose: false,
    customSvgUrl: null,
    customSvgCode: null,
    svgSize: "100px",
    svgColor: "#32c682",
    messageID: "NotiflixLoadingMessage",
    messageFontSize: "20px",
    messageMaxLength: 50,
    messageColor: "#ffffff",
  });
  // var upload = document.getElementById("up");
  // var down = document.getElementById("down");
  // $(upload).removeClass("col-sm-12");
  // $(upload).addClass("col-sm-10");
  // $(down).show();
  Notiflix.Loading.circle(notifyloading);
}
function check(){
  var text = document.getElementById("result-name")
  var a = document.getElementById("url")
  if (text.textContent == "") {
    Notiflix.Notify.failure("Not diagnose...")
  }
  else {
    a.href="/download?id="+text.textContent
  }
}