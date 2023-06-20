(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[5529],{76522:function(e,t,r){"use strict";r.d(t,{Z:function(){return T}});var n=r(59499),s=(r(67294),r(11163)),i=r(41664),a=r.n(i),c=r(47166),o=r.n(c),l=r(27812),d=r(98214),u=r.n(d),p=r(80305),f=r(85893),v=o().bind(u()),_=function(e){var t=e.types,r=void 0===t?[]:t,n=e.activeType,s=void 0===n?null:n,i=e.tags,a=void 0===i?[]:i,c=e.activeTag,o=void 0===c?null:c,d=e.onUpdate,u=[{name:"All Events",value:null}].concat((0,l.Z)(r.map((function(e){var t=e.fields;return{name:t.title,value:t.slug}})))),_=[{name:"All Products / Libraries",value:null}].concat((0,l.Z)(a.map((function(e){var t=e.fields;return{name:t.name,value:t.identifier}}))));return(0,f.jsx)("div",{className:v("root"),children:(0,f.jsxs)("div",{className:v("inner"),children:[(0,f.jsx)("div",{className:v("item"),children:(0,f.jsx)(p.Z,{label:"Event Type",selectedValue:s,options:u,onSelect:function(e){d({type:e,tag:o})},width:245})}),(0,f.jsx)("div",{className:v("item"),children:(0,f.jsx)(p.Z,{label:"Tags",selectedValue:o,options:_,onSelect:function(e){d({type:s,tag:e})},width:245})})]})})};_.defaultProps={};var m=_,h=r(99490),g=r(26970),y=r.n(g),j=r(58199),b=r(88805),O=r(84192);function x(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function P(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?x(Object(r),!0).forEach((function(t){(0,n.Z)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):x(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var w=o().bind(y()),N=function(e){var t=e.slug,r=e.title,n=e.startDate,s=e.ctaText,i=e.ctaLink,c=e.summary,o=e.mainImage,l=e.category,d=(0,O.U6)({route:"/events",startDate:n,slug:t}),u=h.ou.fromISO(n).toFormat("MM . dd . yyyy"),p=h.ou.fromISO(n).toFormat("hh:mm a (ZZZZ)"),v=!!n&&h.ou.fromISO(n).diffNow()<0;c&&c.length>120&&"".concat(c.slice(0,120).trim(),"...");return(0,f.jsx)("div",{className:w("root"),children:(0,f.jsxs)("div",{className:w("inner"),children:[(0,f.jsx)(a(),{href:d,children:(0,f.jsx)("a",{className:w("image"),children:(0,f.jsx)("div",{className:w("cover"),children:o&&(0,f.jsx)(j.Z,P({},o.fields))})})}),(0,f.jsxs)("div",{className:w("content"),children:[l&&(0,f.jsx)("h4",{className:w("category"),children:(0,f.jsx)("a",{href:"/events?type=".concat(l.fields.slug),style:{color:l.fields.tintColor||""},children:l.fields.title})}),(0,f.jsx)("h3",{className:w("title"),children:(0,f.jsx)(a(),{href:d,children:(0,f.jsx)("a",{children:r})})}),!1,(0,f.jsxs)("span",{className:w("date"),children:[u,",\xa0\xa0",p]}),(0,f.jsx)(b.Z,{className:w("button"),href:i||d,text:s||(v?"Watch video":"Register"),hasArrow:!0,align:"left"})]})]})})};N.defaultProps={ctaText:null,ctaLink:null,summary:null,endDate:null,mainImage:null,category:null};var E=N,k=r(86862),D=r(64589),H=r.n(D);function Z(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function F(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?Z(Object(r),!0).forEach((function(t){(0,n.Z)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):Z(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var S=o().bind(H()),I=function(e){var t=e.events,r=e.types,n=e.tags,i=e.activeType,a=e.activeTag,c=e.route,o=e.page,l=e.totalPages,d=e.onUpdateFilters,u=!t||t.length<1,p=(0,s.useRouter)();return(0,f.jsxs)("div",{children:[(0,f.jsx)(m,{types:r,activeType:i,tags:n,activeTag:a,onUpdate:function(e){var t=e.type,r=e.tag,n="/events",s=o<2,i=!1;t&&(n="".concat(n).concat(i?"&":"?","type=").concat(t),i=!0),r&&(n="".concat(n).concat(i?"&":"?","tag=").concat(r),i=!0),s?(p.push(n,void 0,{shallow:s}),d({type:t,tag:r,shallow:s})):window.location=n}}),u&&(0,f.jsx)("div",{className:S("list","list-empty"),children:(0,f.jsx)("div",{className:S("empty"),children:(0,f.jsx)("h2",{children:"No events found"})})}),!u&&(0,f.jsx)("div",{className:S("list"),children:t.map((function(e,t){return(0,f.jsx)("div",{className:S("event"),children:(0,f.jsx)(E,F({},e.fields))},e.sys.id)}))}),(0,f.jsx)(k.Z,{route:c,page:o,totalPages:l,type:i,tag:a})]})};I.defaultProps={events:[],route:"",page:1,totalPages:1,isPast:!1};var T=I},61375:function(e,t,r){"use strict";r.d(t,{Z:function(){return P}});var n=r(59499),s=r(67294),i=r(47166),a=r.n(i),c=r(41664),o=r.n(c),l=r(99490),d=r(41221),u=r.n(d),p=r(58199),f=r(88805),v=r(84192),_=r(85893);function m(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function h(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?m(Object(r),!0).forEach((function(t){(0,n.Z)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):m(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var g=a().bind(u()),y=function(e){var t=e.slug,r=e.title,n=e.startDate,s=e.ctaText,i=e.ctaLink,a=e.summary,c=e.mainImage,d=e.category,u=e.primary,m=(0,v.U6)({route:"/events",startDate:n,slug:t}),y=l.ou.fromISO(n).toFormat("MM . dd . yyyy"),j=l.ou.fromISO(n).toFormat("hh:mm a (ZZZZ)"),b=!!n&&l.ou.fromISO(n).diffNow()<0,O=a&&a.length>120?"".concat(a.slice(0,120).trim(),"..."):a;return(0,_.jsxs)("div",{className:g("event",{primary:u}),children:[u&&(0,_.jsx)(o(),{href:m,children:(0,_.jsx)("a",{className:g("event-image"),tabIndex:-1,children:(0,_.jsx)("div",{className:g("image-wrapper"),children:c&&(0,_.jsx)(p.Z,h(h({},c.fields),{},{sizing:"w=760",description:null}))})})}),(0,_.jsxs)("div",{className:g("event-main"),children:[d&&(0,_.jsx)("h4",{className:g("category"),children:(0,_.jsx)(o(),{href:"/event-category/".concat(d.fields.slug),children:(0,_.jsx)("a",{style:{color:d.fields.tintColor||""},children:d.fields.title})})}),(0,_.jsx)(o(),{href:m,children:(0,_.jsx)("a",{className:g("title"),children:(0,_.jsx)("h3",{children:r})})}),u&&a&&(0,_.jsx)("p",{children:O}),u&&(0,_.jsxs)("span",{className:g("date"),children:[y,",\xa0\xa0",j]}),u&&(0,_.jsx)(f.Z,{className:g("button"),href:i||m,text:s||(b?"Watch video":"Register"),hasArrow:!0,align:"left"})]})]})};function j(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function b(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?j(Object(r),!0).forEach((function(t){(0,n.Z)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):j(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var O=a().bind(u()),x=function(e){var t=e.header,r=e.subheader,n=e.events;if(!n||n.length<1)return null;var i=n[0];return(0,_.jsx)("div",{className:O("container"),children:(0,_.jsxs)("div",{className:O("inner"),children:[(0,_.jsxs)("div",{className:O("header"),children:[(0,_.jsx)("h1",{children:t}),(0,_.jsx)("div",{className:O("spacer")})]}),(0,_.jsxs)("div",{className:O("list"),children:[i&&(0,_.jsx)(y,b(b({},i.fields),{},{primary:!0})),(0,_.jsx)("h3",{className:O("list-header"),children:r}),(0,_.jsx)("div",{className:O("list_inner"),children:n.map((function(e,t){var r=e.fields,n=e.sys;return t>0?(0,s.createElement)(y,b(b({},r),{},{key:n.id,primary:!1})):null}))})]})]})})};x.defaultProps={events:[]};var P=x},13092:function(e,t,r){"use strict";r.d(t,{Z:function(){return k}});var n,s=r(59499),i=r(67294),a=r(41664),c=r.n(a),o=r(47166),l=r.n(o),d=r(9980),u=r.n(d),p=r(58199),f=(r(88805),r(60303)),v=r(586),_=r.n(v);function m(){return m=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var r=arguments[t];for(var n in r)Object.prototype.hasOwnProperty.call(r,n)&&(e[n]=r[n])}return e},m.apply(this,arguments)}var h,g,y=function(e){return i.createElement("svg",m({width:24,height:20,fill:"none",xmlns:"http://www.w3.org/2000/svg",role:"img"},e),n||(n=i.createElement("path",{d:"M1 10h22m0 0-9-9m9 9-9 9",stroke:"#fff",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"})))};function j(){return j=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var r=arguments[t];for(var n in r)Object.prototype.hasOwnProperty.call(r,n)&&(e[n]=r[n])}return e},j.apply(this,arguments)}var b=function(e){return i.createElement("svg",j({width:61,height:61,fill:"none",xmlns:"http://www.w3.org/2000/svg",role:"img"},e),h||(h=i.createElement("path",{opacity:.7,d:"M58.952 30.476c0 15.727-12.75 28.476-28.476 28.476C14.749 58.952 2 46.202 2 30.476 2 14.749 14.75 2 30.476 2c15.727 0 28.476 12.75 28.476 28.476Z",stroke:"#fff",strokeWidth:4})),g||(g=i.createElement("path",{d:"m44.511 29.691-22.045-12.56c-.652-.376-1.466.107-1.466.805v25.12c0 .752.814 1.181 1.466.806l22.045-12.56c.652-.323.652-1.289 0-1.61Z",fill:"#fff"})))};var O=r(66994),x=r(85893);function P(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function w(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?P(Object(r),!0).forEach((function(t){(0,s.Z)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):P(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var N=l().bind(_()),E=function(e){var t=e.header,r=e.body,n=e.ctaText,s=e.ctaLink,a=e.secondaryCtaText,o=e.secondaryCtaLink,l=e.images,d=void 0===l?[]:l,v=e.subsections,_=void 0===v?[]:v,m=e.color,h=void 0===m?"":m,g=e.secondary,j=e.sizing,P=void 0===j?"":j,E=e.page,k=void 0===E?"":E,D=e.styling,H=void 0===D?[]:D,Z=(0,i.useState)(!1),F=Z[0],S=Z[1],I=((0,O.S)().width,d[1]),T=d.length>0?d[0]:null,L=_.filter((function(e){return"embed"!==e.sys.contentType.sys.id}))[0],M=_.filter((function(e){return"embed"===e.sys.contentType.sys.id}))[0],R=["product","pricing"].includes(k),C=(null===L||void 0===L?void 0:L.fields)||{},z=C.body,W=C.ctaText,A=C.ctaLink,U=["about","careers"].includes(k),B=U?s:null,G=U?n:null;d.length>1&&M&&d[1];return(0,x.jsxs)("div",{className:N("wrapper","color_".concat(h)),children:[(0,x.jsxs)("div",{className:N("root",H,"sizing_".concat(P.toLowerCase()),{secondary:g,has_video:M||I,centered:R}),children:[(0,x.jsxs)("div",{className:N("inner"),children:[(0,x.jsxs)("div",{className:N("content"),children:[t&&(0,x.jsx)("h1",{dangerouslySetInnerHTML:{__html:u()({html:!0}).renderInline(t)}}),r&&(0,x.jsx)("div",{dangerouslySetInnerHTML:{__html:u()({html:!0,breaks:!0}).render(r)}}),!U&&n&&s&&(0,x.jsxs)("div",{className:N("links"),children:[(0,x.jsx)(c(),{href:s,children:(0,x.jsx)("a",{className:N("button"),children:n})}),a&&(0,x.jsx)(c(),{href:o,children:(0,x.jsxs)("a",{className:N("button","button_secondary"),onClick:function(){return M?S(!0):null},children:[a,!1]})})]}),W&&!(null!==A&&void 0!==A&&A.includes("ray-summit-2022"))&&(0,x.jsx)(c(),{href:A||"",children:(0,x.jsxs)("a",{className:N("full_cta"),children:[z&&(0,x.jsx)("span",{children:z}),W,(0,x.jsx)(y,{})]})}),W&&(null===A||void 0===A?void 0:A.includes("ray-summit-2022"))&&(0,x.jsxs)("a",{href:A,className:N("full_cta"),children:[z&&(0,x.jsx)("span",{children:z}),W,(0,x.jsx)(y,{})]}),B&&(0,x.jsx)(c(),{href:B,children:(0,x.jsx)("a",{className:N("button"),children:G})})]}),!R&&M&&(0,x.jsx)("div",{className:N("video-preview"),children:(0,x.jsx)("div",{className:N("frame"),onClick:function(){return S(!0)},children:(0,x.jsxs)("div",{className:N("preview-inner"),children:[M.fields.previewImage&&(0,x.jsx)(p.Z,w({},M.fields.previewImage.fields)),(0,x.jsx)("div",{className:N("play-button"),children:(0,x.jsx)(b,{})})]})})}),!M&&I&&(0,x.jsx)("div",{className:N("inline-image"),children:(0,x.jsx)("div",{className:N("frame"),children:(0,x.jsx)("div",{className:N("preview-inner"),children:(0,x.jsx)(p.Z,w({},I.fields))})})})]}),T&&(0,x.jsx)("div",{className:N("background"),style:{backgroundPosition:"right top",backgroundRepeat:"no-repeat",backgroundSize:"712px 712px",backgroundImage:"The Ray Ecosystem"!==t?'url("https://images.ctfassets.net/xjan103pcp94/543nAUbkQJdvxVD20a5WDR/2253f2cb701d28f2dbd4ec3283c6cafd/hero-pattern_2x.png")':""}})]}),(0,x.jsx)("div",{className:N("The Ray Ecosystem"===t?"trapezia_grey":"trapezia")}),M&&(0,x.jsx)(f.Z,w(w({},M.fields),{},{isOpen:F,closeModal:function(){return S(!1)}}))]})};E.defaultProps={header:null,body:null,ctaText:null,ctaLink:null,secondary:!1};var k=E},26629:function(e,t,r){"use strict";r(67294);var n=r(47166),s=r.n(n),i=r(26555),a=r.n(i),c=r(85893),o=s().bind(a());t.Z=function(e){var t=e.title;return(0,c.jsx)("div",{className:o("container"),children:(0,c.jsx)("div",{className:o("inner"),children:(0,c.jsxs)("div",{className:o("header"),children:[(0,c.jsx)("h1",{children:t}),(0,c.jsx)("div",{className:o("spacer")})]})})})}},90606:function(e,t,r){"use strict";r.d(t,{H5:function(){return o},Wo:function(){return a},dz:function(){return l},pi:function(){return c}});var n=r(59499);function s(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?s(Object(r),!0).forEach((function(t){(0,n.Z)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):s(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var a=18,c=18,o=function(e){var t=e.page,r=e.type,n=e.allTypes,s=e.tag,c=e.allTags,o={skip:t>1?(t-2)*a+12:0,limit:1===t?12:a,order:"-fields.publishedDate,sys.id"};if(r){var l=n.filter((function(e){return e.fields.slug===r}))[0];o=i(i({},o),{},{"fields.type.sys.id":l.sys.id})}if(s){var d=c.filter((function(e){return e.fields.identifier===s}))[0];o=i(i({},o),{},{"fields.tags.sys.id":d.sys.id})}return o},l=function(e){var t=e.page,r=e.type,n=e.allTypes,s=e.tag,a=e.allTags,o={skip:t>1?(t-2)*c+12:0,limit:1===t?12:c,order:"-fields.startDate,sys.id"};if(r){var l=n.filter((function(e){return e.fields.slug===r}))[0];o=i(i({},o),{},{"fields.category.sys.id":l.sys.id})}if(s){var d=a.filter((function(e){return e.fields.identifier===s}))[0];o=i(i({},o),{},{"fields.tags.sys.id":d.sys.id})}return o}},98214:function(e){e.exports={root:"EventFilters_root__qCFDE",inner:"EventFilters_inner__eioYo",item:"EventFilters_item__HjzzC"}},64589:function(e){e.exports={list:"EventList_list__j2VdM","list-empty":"EventList_list-empty__5r1hV",empty:"EventList_empty__rM_rT"}},26970:function(e){e.exports={root:"EventListItem_root__YRr69",image:"EventListItem_image__6kDEl",cover:"EventListItem_cover__A5CD2",content:"EventListItem_content__DFNMm",title:"EventListItem_title__MdlnP",category:"EventListItem_category__aWHpG",button:"EventListItem_button__Beml5",date:"EventListItem_date__bR0xC"}},41221:function(e){e.exports={container:"FeaturedEvents_container__6umHg",inner:"FeaturedEvents_inner__XvFcA",header:"FeaturedEvents_header__qmhIG",spacer:"FeaturedEvents_spacer__jxAgb","list-header":"FeaturedEvents_list-header__vPPqZ",list_inner:"FeaturedEvents_list_inner__vKOMg",event:"FeaturedEvents_event__opaZT",primary:"FeaturedEvents_primary__RFgJ2","event-image":"FeaturedEvents_event-image__bOROY","image-wrapper":"FeaturedEvents_image-wrapper__KWUSu","event-main":"FeaturedEvents_event-main__quoWo",category:"FeaturedEvents_category__I_kYS",date:"FeaturedEvents_date__M8FDl",title:"FeaturedEvents_title__4vJaB",button:"FeaturedEvents_button__f5MPF"}},586:function(e){e.exports={wrapper:"PageHero_wrapper__lPRYV",root:"PageHero_root__b4B00",padded:"PageHero_padded__uapuR",color_light:"PageHero_color_light__GWBpF",background:"PageHero_background__ItwlU",trapezia:"PageHero_trapezia__jx0jI",trapezia_grey:"PageHero_trapezia_grey__8rFVo",inner:"PageHero_inner__ICR6N","transparent-header":"PageHero_transparent-header__6wIfJ",secondary:"PageHero_secondary__4ePa4",has_video:"PageHero_has_video__ZfeiG",centered:"PageHero_centered__q4sRh",sizing_small:"PageHero_sizing_small__MByYo",content:"PageHero_content__ZukGq",button:"PageHero_button__Q2LC_",links:"PageHero_links__dpKXr",button_secondary:"PageHero_button_secondary__79R5W",full_cta:"PageHero_full_cta__4MlN7","inline-image":"PageHero_inline-image__RUXBj","video-preview":"PageHero_video-preview__iEasr",frame:"PageHero_frame__5uaAI","preview-inner":"PageHero_preview-inner__D7lx_","play-button":"PageHero_play-button__fK9eH",cta:"PageHero_cta__BGetn"}},26555:function(e){e.exports={container:"SimpleHero_container__OP08F",inner:"SimpleHero_inner__Ek_r_",header:"SimpleHero_header__HmRWX",spacer:"SimpleHero_spacer__QSrcG"}}}]);