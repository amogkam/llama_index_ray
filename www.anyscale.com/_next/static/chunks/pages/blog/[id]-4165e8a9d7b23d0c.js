(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[5610],{25948:function(e,t,n){"use strict";var r=n(67294),i=n(47166),s=n.n(i),a=n(99490),o=n(25949),c=n.n(o),l=n(85893),u=s().bind(c());t.Z=function(e){var t=e.authors,n=e.publishedDate,i=(e.tags,n?a.ou.fromISO(n).toFormat("MMMM d, yyyy"):null);return(0,l.jsxs)("div",{className:u("container"),children:[t&&(0,l.jsxs)("span",{className:u("authors"),children:[(0,l.jsx)("span",{children:"By "}),t.map((function(e,n){return(0,l.jsxs)(r.Fragment,{children:[(0,l.jsx)("a",{href:"/blog?author=".concat(e.fields.slug),children:e.fields.name}),n<t.length-2&&(0,l.jsx)("span",{children:", "}),n===t.length-2&&(0,l.jsx)("span",{children:" and "})]},e.sys.id)})),i&&(0,l.jsx)("span",{children:"\xa0\xa0\xa0"})]}),(0,l.jsx)("span",{className:u("article-published-tags"),children:i&&(0,l.jsxs)("span",{className:u("article-published"),children:["|\xa0\xa0\xa0",i]})})]})}},61212:function(e,t,n){"use strict";var r=n(92777),i=n(82262),a=n(10748),o=n(45959),c=n(63553),l=n(37247),u=n(59499),d=n(67294),h=n(47166),m=n.n(h),f=n(49074),p=n.n(f),v=n(85893);function g(e){var t=function(){if("undefined"===typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"===typeof Proxy)return!0;try{return Boolean.prototype.valueOf.call(Reflect.construct(Boolean,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,r=(0,l.Z)(e);if(t){var i=(0,l.Z)(this).constructor;n=Reflect.construct(r,arguments,i)}else n=r.apply(this,arguments);return(0,c.Z)(this,n)}}var b=m().bind(p()),_=function(e){(0,o.Z)(n,e);var t=g(n);function n(){var e;(0,r.Z)(this,n);for(var i=arguments.length,s=new Array(i),o=0;o<i;o++)s[o]=arguments[o];return e=t.call.apply(t,[this].concat(s)),(0,u.Z)((0,a.Z)(e),"state",{formData:{},loading:!1,success:!1,error:null,focused:!1}),(0,u.Z)((0,a.Z)(e),"handleSuccess",(function(){e.setState({loading:!1,success:!0})})),e}return(0,i.Z)(n,[{key:"componentDidMount",value:function(){var e=this,t=this.props.formId,n=document.createElement("script");n.src="https://js.hsforms.net/forms/v2.js",document.body.appendChild(n),n.addEventListener("load",(function(){window.hbspt&&window.hbspt.forms.create({portalId:"20523749",formId:t,target:"#hubspotForm",onFormSubmit:function(t){e.handleSuccess()}})})),window.jQuery=window.jQuery||function(e){return"string"==typeof e?document.querySelector(s):e}}},{key:"render",value:function(){var e=this.props,t=(e.formId,e.condensed),n=this.state,r=(n.loading,n.error,n.success);n.focused;return(0,v.jsxs)("div",{className:b("root",{condensed:t}),children:[(0,v.jsx)("div",{className:b("form",{hidden:r}),children:(0,v.jsx)("div",{id:"hubspotForm"})}),r&&(0,v.jsx)("div",{className:b("success-message"),children:"Thank you for signing up!"})]})}}]),n}(d.Component);_.defaultProps={ctaText:"Enter email",listId:null},t.Z=_},94383:function(e,t,n){"use strict";n.r(t),n.d(t,{__N_SSG:function(){return pe},default:function(){return ve}});var r=n(67294),i=n(47166),s=n.n(i),a=n(11163),o=n(8423),c=n(92477),l=n(59499),u=n(9980),d=n.n(u),h=n(92956),m=n.n(h),f=n(58199),p=n(25948),v=n(85893);function g(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function b(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?g(Object(n),!0).forEach((function(t){(0,l.Z)(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):g(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}var _=s().bind(m()),j=function(e){var t=e.title,n=e.authors,r=e.publishedDate,i=e.intro,s=e.mainImage,a=e.mainImageFit,o=void 0===a?"contain":a,c=e.showMainImage,l=void 0!==c&&c,u=e.hideIntro,h=void 0!==u&&u;return(0,v.jsx)("div",{className:_("container"),children:(0,v.jsxs)("div",{className:_("inner"),children:[(0,v.jsx)("h1",{className:_("title"),children:t}),(0,v.jsx)("div",{className:_("details"),children:(0,v.jsx)(p.Z,{authors:n,publishedDate:r})}),(l||!h)&&(0,v.jsxs)("div",{className:_("image-intro"),children:[s&&l&&(0,v.jsx)("div",{className:_("main-image"),children:(0,v.jsx)(f.Z,b(b({},s.fields),{},{contain:"cover"!==o,sizing:"w=1200"}))}),i&&!h&&(0,v.jsx)("div",{className:_("intro"),dangerouslySetInnerHTML:{__html:d()({html:!0,breaks:!0}).render(i)}})]})]})})};j.defaultProps={authors:null,tags:null,intro:null,mainImage:null};var w,x=j,y=n(24684),O=n(92777),Z=n(82262),N=n(10748),E=n(45959),k=n(63553),I=n(37247),P=n(80717),H=n.n(P),D=n(61212);function C(){return C=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},C.apply(this,arguments)}var S,F=function(e){return r.createElement("svg",C({width:24,height:24,fill:"none",xmlns:"http://www.w3.org/2000/svg",role:"img"},e),w||(w=r.createElement("path",{d:"M24 12c0-6.628-5.372-12-12-12S0 5.372 0 12c0 5.99 4.388 10.955 10.125 11.855v-8.386H7.078V12h3.047V9.356c0-3.007 1.79-4.668 4.533-4.668 1.312 0 2.686.234 2.686.234v2.953H15.83c-1.49 0-1.955.926-1.955 1.875V12h3.328l-.532 3.469h-2.796v8.386C19.613 22.955 24 17.99 24 12Z",fill:"#999"})))};function R(){return R=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},R.apply(this,arguments)}var M,U=function(e){return r.createElement("svg",R({width:24,height:24,fill:"none",xmlns:"http://www.w3.org/2000/svg",role:"img"},e),S||(S=r.createElement("path",{d:"M24 12c0-6.628-5.372-12-12-12S0 5.372 0 12c0 5.99 4.388 10.955 10.125 11.855v-8.386H7.078V12h3.047V9.356c0-3.007 1.79-4.668 4.533-4.668 1.312 0 2.686.234 2.686.234v2.953H15.83c-1.49 0-1.955.926-1.955 1.875V12h3.328l-.532 3.469h-2.796v8.386C19.613 22.955 24 17.99 24 12Z",fill:"#234999"})))};function T(){return T=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},T.apply(this,arguments)}var A,B=function(e){return r.createElement("svg",T({width:24,height:20,fill:"none",xmlns:"http://www.w3.org/2000/svg",role:"img"},e),M||(M=r.createElement("path",{d:"M24 2.436c-.955.358-1.791.597-2.866.716 1.075-.597 1.791-1.552 2.15-2.746-.956.597-2.03.955-3.105 1.194-1.79-2.03-4.895-2.15-6.925-.239-1.314 1.194-1.91 2.985-1.433 4.776C7.88 5.9 4.299 3.988 1.79 1.003a4.305 4.305 0 0 0-.716 2.388c0 1.672.835 3.105 2.149 4.06-.717 0-1.433-.12-2.15-.597v.12c0 2.268 1.672 4.298 3.94 4.775-.477.12-.835.24-1.313.24-.358 0-.597 0-.955-.12.597 2.03 2.508 3.343 4.538 3.343-1.672 1.313-3.821 2.15-6.09 2.15-.358 0-.836 0-1.194-.12 2.269 1.433 4.896 2.268 7.522 2.268 7.642.12 13.851-6.089 13.97-13.73v-.837c1.075-.716 1.911-1.552 2.508-2.507Z",fill:"#999"})))};function L(){return L=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},L.apply(this,arguments)}var V,X=function(e){return r.createElement("svg",L({width:24,height:20,fill:"none",xmlns:"http://www.w3.org/2000/svg",role:"img"},e),A||(A=r.createElement("path",{d:"M24 2.436c-.955.358-1.791.597-2.866.716 1.075-.597 1.791-1.552 2.15-2.746-.956.597-2.03.955-3.105 1.194-1.79-2.03-4.895-2.15-6.925-.239-1.314 1.194-1.91 2.985-1.433 4.776C7.88 5.9 4.299 3.988 1.79 1.003a4.305 4.305 0 0 0-.716 2.388c0 1.672.835 3.105 2.149 4.06-.717 0-1.433-.12-2.15-.597v.12c0 2.268 1.672 4.298 3.94 4.775-.477.12-.835.24-1.313.24-.358 0-.597 0-.955-.12.597 2.03 2.508 3.343 4.538 3.343-1.672 1.313-3.821 2.15-6.09 2.15-.358 0-.836 0-1.194-.12 2.269 1.433 4.896 2.268 7.522 2.268 7.642.12 13.851-6.089 13.97-13.73v-.837c1.075-.716 1.911-1.552 2.508-2.507Z",fill:"#234999"})))};function Q(){return Q=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Q.apply(this,arguments)}var q,J=function(e){return r.createElement("svg",Q({width:24,height:25,fill:"none",xmlns:"http://www.w3.org/2000/svg",role:"img"},e),V||(V=r.createElement("path",{d:"M22.191 0H1.809C.844 0 0 .724 0 1.688v20.624c0 1.085.844 1.809 1.809 1.809h20.382c.965 0 1.809-.724 1.809-1.689V1.688C24 .724 23.156 0 22.191 0ZM7.236 20.14H3.618V9.287h3.618v10.855ZM5.427 7.84c-.965 0-1.809-.845-1.809-1.93 0-1.086.844-1.93 1.93-1.93 1.085-.12 1.93.603 2.05 1.688.12 1.086-.724 2.05-1.688 2.171h-.483Zm14.955 12.3h-3.618v-5.788c0-1.447-.483-2.412-1.81-2.412-.843 0-1.567.482-1.808 1.326-.12.242-.12.603-.12.845v6.03H9.406V9.286h3.618v1.568c.724-1.206 1.93-1.809 3.256-1.809 2.413 0 4.222 1.568 4.222 4.945v6.15h-.121Z",fill:"#999"})))};function W(){return W=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},W.apply(this,arguments)}var G=function(e){return r.createElement("svg",W({width:24,height:25,fill:"none",xmlns:"http://www.w3.org/2000/svg",role:"img"},e),q||(q=r.createElement("path",{d:"M22.191 0H1.809C.844 0 0 .724 0 1.688v20.624c0 1.085.844 1.809 1.809 1.809h20.382c.965 0 1.809-.724 1.809-1.689V1.688C24 .724 23.156 0 22.191 0ZM7.236 20.14H3.618V9.287h3.618v10.855ZM5.427 7.84c-.965 0-1.809-.845-1.809-1.93 0-1.086.844-1.93 1.93-1.93 1.085-.12 1.93.603 2.05 1.688.12 1.086-.724 2.05-1.688 2.171h-.483Zm14.955 12.3h-3.618v-5.788c0-1.447-.483-2.412-1.81-2.412-.843 0-1.567.482-1.808 1.326-.12.242-.12.603-.12.845v6.03H9.406V9.286h3.618v1.568c.724-1.206 1.93-1.809 3.256-1.809 2.413 0 4.222 1.568 4.222 4.945v6.15h-.121Z",fill:"#234999"})))};function K(e){var t=function(){if("undefined"===typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"===typeof Proxy)return!0;try{return Boolean.prototype.valueOf.call(Reflect.construct(Boolean,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,r=(0,I.Z)(e);if(t){var i=(0,I.Z)(this).constructor;n=Reflect.construct(r,arguments,i)}else n=r.apply(this,arguments);return(0,k.Z)(this,n)}}var Y=s().bind(H()),z=function(e){(0,E.Z)(n,e);var t=K(n);function n(){var e;(0,O.Z)(this,n);for(var r=arguments.length,i=new Array(r),s=0;s<r;s++)i[s]=arguments[s];return e=t.call.apply(t,[this].concat(i)),(0,l.Z)((0,N.Z)(e),"state",{facebookUrl:"",twitterUrl:"",linkedinUrl:""}),(0,l.Z)((0,N.Z)(e),"shareFacebookUrl",(function(){var e=encodeURIComponent(window.location.href.split("?")[0]);return"https://www.facebook.com/sharer/sharer.php?u=".concat(e)})),(0,l.Z)((0,N.Z)(e),"shareTwitterUrl",(function(){var t=e.props.title,n=encodeURIComponent(window.location.href.split("?")[0]);return"https://www.twitter.com/share?text=".concat(encodeURIComponent(t),"&url=").concat(n,"&via=anyscalecompute")})),(0,l.Z)((0,N.Z)(e),"shareLinkedinUrl",(function(){var e=encodeURIComponent(window.location.href.split("?")[0]);return"https://www.linkedin.com/sharing/share-offsite/?url=".concat(e)})),e}return(0,Z.Z)(n,[{key:"componentDidMount",value:function(){this.setState({facebookUrl:this.shareFacebookUrl(),twitterUrl:this.shareTwitterUrl(),linkedinUrl:this.shareLinkedinUrl()})}},{key:"render",value:function(){var e=this.props.tags,t=this.state,n=t.facebookUrl,i=t.twitterUrl,s=t.linkedinUrl;return(0,v.jsxs)("div",{className:Y("root"),children:[(0,v.jsx)("h4",{className:Y("label"),children:"Sharing"}),(0,v.jsxs)("div",{className:Y("sharing","section"),children:[(0,v.jsxs)("a",{target:"_blank",rel:"noreferrer",href:n,"aria-label":"Share on Facebook",children:[(0,v.jsx)(F,{}),(0,v.jsx)("div",{className:Y("active"),children:(0,v.jsx)(U,{})})]}),(0,v.jsxs)("a",{target:"_blank",rel:"noreferrer",href:i,"aria-label":"Share on Twitter",children:[(0,v.jsx)(B,{}),(0,v.jsx)("div",{className:Y("active"),children:(0,v.jsx)(X,{})})]}),(0,v.jsxs)("a",{target:"_blank",rel:"noreferrer",href:s,"aria-label":"Share on Linkedin",children:[(0,v.jsx)(J,{}),(0,v.jsx)("div",{className:Y("active"),children:(0,v.jsx)(G,{})})]})]}),e&&(0,v.jsxs)("div",{className:Y("tags-wrapper","section"),children:[(0,v.jsx)("h4",{className:Y("label"),children:"Tags"}),(0,v.jsx)("span",{className:Y("tags"),children:e.map((function(e,t){return(0,v.jsxs)(r.Fragment,{children:[t>0&&(0,v.jsx)("span",{children:", "}),(0,v.jsx)("span",{className:Y("tag"),children:e.fields.name})]},e.sys.id)}))})]}),(0,v.jsxs)("div",{className:Y("form-wrapper"),children:[(0,v.jsx)("h4",{className:Y("label"),children:"Sign up for product updates"}),(0,v.jsx)(D.Z,{ctaText:"Subscribe",formId:"3571cddd-036f-4fc4-a07c-b760cb08df70",condensed:!0})]})]})}}]),n}(r.PureComponent);z.defaultProps={title:"",tags:null};var $,ee=z,te=n(41664),ne=n.n(te);function re(){return re=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},re.apply(this,arguments)}var ie=function(e){return r.createElement("svg",re({width:15,height:12,fill:"none",xmlns:"http://www.w3.org/2000/svg",role:"img"},e),$||($=r.createElement("path",{d:"M1 6h12.222m0 0-5-5m5 5-5 5",stroke:"#234999",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"})))},se=n(40965),ae=n.n(se);function oe(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function ce(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?oe(Object(n),!0).forEach((function(t){(0,l.Z)(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):oe(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}var le=s().bind(ae()),ue=function(e){var t=e.recommendations,n=void 0===t?[]:t;return(0,v.jsxs)("div",{className:le("root"),children:[(0,v.jsx)("h4",{className:le("header"),children:"Recommended content"}),n.map((function(e){var t,n=(null===e||void 0===e?void 0:e.fields)||{},r=n.content,i=n.title,s=n.ctaText,a=n.thumbnail,o=n.url,c="/",l=null===r||void 0===r?void 0:r.sys.contentType.sys.id;"event"===l?(c="/events/",t=r.fields.thumbnail||r.fields.mainImage):"blogPost"===l?(c="/blog/",t=r.fields.thumbnail||r.fields.mainImage):"summit"===l&&(c="/",t=r.fields.thumbnail||r.fields.shareImage);var u=a||t,d=r?"".concat(c).concat(r.fields.slug):o,h=i||r.fields.title,m=s||"Read more";return(0,v.jsx)(ne(),{href:d,children:(0,v.jsxs)("a",{className:le("item"),children:[(0,v.jsx)("div",{className:le("thumbnail"),children:u&&(0,v.jsx)(f.Z,ce({},u.fields))}),(0,v.jsxs)("div",{className:le("item-content"),children:[(0,v.jsx)("h4",{children:h}),(0,v.jsxs)("span",{children:[m,(0,v.jsx)(ie,{})]})]})]})},r.sys.id)}))]})},de=n(35921),he=n.n(de),me=s().bind(he()),fe=function(e){var t=e.site,n=e.title,r=e.seoTitle,i=e.canonical,s=e.description,l=e.mainImage,u=e.body,d=e.publishedDate,h=e.authors,m=e.intro,f=e.tags,p=e.hideIntro,g=e.showMainImage,b=e.mainImageFit,_=e.bannerText,j=e.bannerCta,w=e.bannerLink,O=e.recommendations;if((0,a.useRouter)().isFallback)return(0,v.jsx)("div",{children:"Loading..."});var Z=!!_;return(0,v.jsx)(o.Z,{site:t,title:n,seoTitle:r,description:s,shareImage:l,canonical:i,bannerText:_,bannerCta:j,bannerLink:w,showBanner:Z,children:(0,v.jsxs)("div",{className:me("root"),children:[(0,v.jsx)(c.Z,{parentFolderName:"Blog",parentFolderLink:"/blog",folderName:"Blog Detail"}),(0,v.jsxs)("div",{className:me("inner"),children:[(0,v.jsxs)("div",{className:me("main"),children:[(0,v.jsx)(x,{title:n,authors:h,publishedDate:d,mainImage:l,mainImageFit:b,intro:m,hideIntro:p,showMainImage:g}),(0,v.jsx)(y.Z,{body:u,page:"article"})]}),(0,v.jsxs)("div",{className:me("aside"),children:[(0,v.jsx)(ee,{title:n,tags:f}),(0,v.jsx)(ue,{recommendations:O})]})]})]})})};fe.defaultProps={description:null,publishedDate:null,mainImage:null,authors:[],bannerText:null,bannerLink:null,canonical:null,recommendations:[]};var pe=!0,ve=fe},40516:function(e,t,n){(window.__NEXT_P=window.__NEXT_P||[]).push(["/blog/[id]",function(){return n(94383)}])},25949:function(e){e.exports={authors:"ArticleDetails_authors__aqDWy","article-published-tags":"ArticleDetails_article-published-tags__9VxRY"}},80717:function(e){e.exports={label:"ArticleExtras_label__JQEEO",sharing:"ArticleExtras_sharing__LSXs1",active:"ArticleExtras_active__cv689",section:"ArticleExtras_section__26jYL",tags:"ArticleExtras_tags__Q_xZs","form-wrapper":"ArticleExtras_form-wrapper__rTr1C"}},92956:function(e){e.exports={inner:"ArticleHero_inner__lg98I",title:"ArticleHero_title__JB2va","image-intro":"ArticleHero_image-intro__mj4hS","main-image":"ArticleHero_main-image__Jug5S",intro:"ArticleHero_intro__cVIql",details:"ArticleHero_details__DZgPt"}},49074:function(e){e.exports={form:"HubspotEmailForm_form___3bhH",root:"HubspotEmailForm_root__Rwc_O",hidden:"HubspotEmailForm_hidden__9pnHI",fadeIn:"HubspotEmailForm_fadeIn___tNXb",condensed:"HubspotEmailForm_condensed___nNtN",message:"HubspotEmailForm_message__3uOXE",active:"HubspotEmailForm_active__EZRK_","success-message":"HubspotEmailForm_success-message__I4dDH",fadeInUp:"HubspotEmailForm_fadeInUp__5b5G4"}},40965:function(e){e.exports={header:"RecommendedContent_header__3Cuyf",item:"RecommendedContent_item__e9DDs",thumbnail:"RecommendedContent_thumbnail__bD3bd","item-content":"RecommendedContent_item-content__rXTae"}},35921:function(e){e.exports={root:"BlogPost_root__ly_um",inner:"BlogPost_inner__70rgb",main:"BlogPost_main__husly",aside:"BlogPost_aside__BK_Wk"}}},function(e){e.O(0,[9809,1812,9065,9614,8423,7846,9774,2888,179],(function(){return t=40516,e(e.s=t);var t}));var t=e.O();_N_E=t}]);