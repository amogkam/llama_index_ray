(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[5165],{99651:function(e,t,r){"use strict";var n=r(33227),i=r(88361),o=r(85971),a=r(52715),l=r(91193);function s(e){var t=function(){if("undefined"===typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"===typeof Proxy)return!0;try{return Boolean.prototype.valueOf.call(Reflect.construct(Boolean,[],(function(){}))),!0}catch(e){return!1}}();return function(){var r,n=l(e);if(t){var i=l(this).constructor;r=Reflect.construct(n,arguments,i)}else r=n.apply(this,arguments);return a(this,r)}}Object.defineProperty(t,"__esModule",{value:!0}),t.default=void 0;var u=r(92648).Z,c=u(r(67294)),d=u(r(72717)),f={400:"Bad Request",404:"This page could not be found",405:"Method Not Allowed",500:"Internal Server Error"};function p(e){var t=e.res,r=e.err;return{statusCode:t&&t.statusCode?t.statusCode:r?r.statusCode:404}}var h={error:{fontFamily:'-apple-system, BlinkMacSystemFont, Roboto, "Segoe UI", "Fira Sans", Avenir, "Helvetica Neue", "Lucida Grande", sans-serif',height:"100vh",textAlign:"center",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center"},desc:{display:"inline-block",textAlign:"left",lineHeight:"49px",height:"49px",verticalAlign:"middle"},h1:{display:"inline-block",margin:0,marginRight:"20px",padding:"0 23px 0 0",fontSize:"24px",fontWeight:500,verticalAlign:"top",lineHeight:"49px"},h2:{fontSize:"14px",fontWeight:"normal",lineHeight:"49px",margin:0,padding:0}},g=function(e){o(r,e);var t=s(r);function r(){return n(this,r),t.apply(this,arguments)}return i(r,[{key:"render",value:function(){var e=this.props,t=e.statusCode,r=e.withDarkMode,n=void 0===r||r,i=this.props.title||f[t]||"An unexpected error has occurred";return c.default.createElement("div",{style:h.error},c.default.createElement(d.default,null,c.default.createElement("title",null,t?"".concat(t,": ").concat(i):"Application error: a client-side exception has occurred")),c.default.createElement("div",null,c.default.createElement("style",{dangerouslySetInnerHTML:{__html:"\n                body { margin: 0; color: #000; background: #fff; }\n                .next-error-h1 {\n                  border-right: 1px solid rgba(0, 0, 0, .3);\n                }\n\n                ".concat(n?"@media (prefers-color-scheme: dark) {\n                  body { color: #fff; background: #000; }\n                  .next-error-h1 {\n                    border-right: 1px solid rgba(255, 255, 255, .3);\n                  }\n                }":"")}}),t?c.default.createElement("h1",{className:"next-error-h1",style:h.h1},t):null,c.default.createElement("div",{style:h.desc},c.default.createElement("h2",{style:h.h2},this.props.title||t?i:c.default.createElement(c.default.Fragment,null,"Application error: a client-side exception has occurred (see the browser console for more information)"),"."))))}}]),r}(c.default.Component);g.displayName="ErrorPage",g.getInitialProps=p,g.origGetInitialProps=p,t.default=g},96547:function(e,t,r){"use strict";r.r(t),r.d(t,{__N_SSG:function(){return a}});r(67294);var n=r(12918),i=r(8423),o=r(85893),a=!0;t.default=function(e){var t=e.site;return(0,o.jsx)(i.Z,{site:t,children:(0,o.jsx)("div",{style:{marginTop:"-100px",position:"relative",zIndex:"-1"},children:(0,o.jsx)(n.default,{statusCode:404})})})}},23967:function(e,t,r){"use strict";r.r(t),r.d(t,{__N_SSG:function(){return l}});r(67294);var n=r(11163),i=r(2007),o=r(17689),a=(r(96547),r(85893)),l=!0;t.default=(0,n.withRouter)((function(e){var t,r=e.router,n=e.site,l=e.slug,s=(e.summitSlug,e.title),u=e.navItems,c=e.headerLogo,d=e.headerLogoWhite,f=e.footer,p=e.seoTitle,h=e.description,g=e.registerText,m=e.registerUrl,v=e.shareImage,x=e.sections,y=(e.mainSponsors,e.agendaData);return r&&r.isFallback?(0,a.jsx)("div",{children:"Loading..."}):!x||x.length<1?(console.log("Error loading Summmit page: [sections]",x),r.push("/ray-summit-2022"),null):(0,a.jsx)(i.Z,{site:n,navItems:u,headerLogo:c,headerLogoWhite:d,footer:f,title:s,seoTitle:p,description:h,shareImage:v,registerText:g,registerUrl:m,transparentHeader:null===(t=x[0].fields.styling)||void 0===t?void 0:t.includes("transparent-header"),children:x.map((function(e,t){return(0,a.jsx)(o.Z,{section:e,index:t,agendaData:y,page:l},e.sys.id)}))})}))},62765:function(e,t,r){(window.__NEXT_P=window.__NEXT_P||[]).push(["/ray-summit-2022/[id]",function(){return r(23967)}])},12918:function(e,t,r){e.exports=r(99651)}},function(e){e.O(0,[9774,9809,1812,9065,6283,3093,3829,8423,4273,2888,179],(function(){return t=62765,e(e.s=t);var t}));var t=e.O();_N_E=t}]);