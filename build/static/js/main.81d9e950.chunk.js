(this.webpackJsonpclient=this.webpackJsonpclient||[]).push([[0],{38:function(e,c,t){},39:function(e,c,t){"use strict";t.r(c);var s=t(2),a=t.n(s),n=t(13),l=t.n(n),i=t(3),r=t(14),j=t.n(r),o=t(0),m=function(){var e=Object(s.useState)("/pic"),c=Object(i.a)(e,2),t=c[0],a=c[1],n=Object(s.useState)("headphone"),l=Object(i.a)(n,2),r=l[0],m=l[1],u=Object(s.useState)(Date.now()),b=Object(i.a)(u,2),d=b[0],O=b[1],g=Object(s.useState)(""),h=Object(i.a)(g,2),x=h[0],p=h[1],f=Object(s.useRef)(null);return Object(s.useEffect)((function(){j.a.get("https://api.unsplash.com/search/photos?query=".concat(r,"&client_id=").concat("m8C8xJeSnEdq2PCiTs25yUyslDd7yRmdmj3_8LpAEuA")).then((function(e){console.log(e.data),p(e.data)}))}),[r]),Object(o.jsxs)(o.Fragment,{children:[Object(o.jsx)("div",{children:Object(o.jsxs)("form",{onSubmit:function(e){e.preventDefault();var c=new FormData;c.append("file",f.files[0]),fetch("/upload",{method:"POST",body:c}).then((function(e){e.json().then((function(e){a(e.picture),O(Date.now()),m(e.result)}))}))},children:[Object(o.jsx)("input",{type:"file",ref:function(e){f=e},id:"input-file",name:"input-file",accept:"image/*"}),Object(o.jsx)("div",{children:Object(o.jsx)("button",{className:"btn-upload",children:"Classify"})}),Object(o.jsx)("div",{class:"preview-box",children:Object(o.jsx)("img",{src:"".concat(t,"?").concat(d)})}),Object(o.jsx)("p",{className:"heading",children:r&&"That's a ".concat(r)})]})}),Object(o.jsx)("div",{className:"image_results",children:Object(o.jsxs)(o.Fragment,{children:[x&&Object(o.jsx)("img",{className:"image",src:"".concat(x.results[0].urls.small,"?").concat(d)}),x&&Object(o.jsx)("img",{className:"image",src:"".concat(x.results[1].urls.small,"?").concat(d)}),x&&Object(o.jsx)("img",{className:"image",src:"".concat(x.results[2].urls.small,"?").concat(d)}),x&&Object(o.jsx)("img",{className:"image",src:"".concat(x.results[3].urls.small,"?").concat(d)}),x&&Object(o.jsx)("img",{className:"image",src:"".concat(x.results[4].urls.small,"?").concat(d)}),x&&Object(o.jsx)("img",{className:"image",src:"".concat(x.results[5].urls.small,"?").concat(d)}),x&&Object(o.jsx)("img",{className:"image",src:"".concat(x.results[6].urls.small,"?").concat(d)}),x&&Object(o.jsx)("img",{className:"image",src:"".concat(x.results[7].urls.small,"?").concat(d)}),x&&Object(o.jsx)("img",{className:"image",src:"".concat(x.results[8].urls.small,"?").concat(d)}),x&&Object(o.jsx)("img",{className:"image",src:"".concat(x.results[9].urls.small,"?").concat(d)})]})})]})};t(38);var u=function(){return Object(o.jsxs)("div",{className:"App",children:[Object(o.jsx)("p",{className:"heading",children:"Image Predictor"}),Object(o.jsx)("div",{class:"container",children:Object(o.jsx)(m,{})})]})},b=function(e){e&&e instanceof Function&&t.e(3).then(t.bind(null,40)).then((function(c){var t=c.getCLS,s=c.getFID,a=c.getFCP,n=c.getLCP,l=c.getTTFB;t(e),s(e),a(e),n(e),l(e)}))};l.a.render(Object(o.jsx)(a.a.StrictMode,{children:Object(o.jsx)(u,{})}),document.getElementById("root")),b()}},[[39,1,2]]]);
//# sourceMappingURL=main.81d9e950.chunk.js.map