(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[9027],{25948:function(e,t,r){"use strict";var n=r(67294),i=r(47166),a=r.n(i),s=r(99490),c=r(25949),l=r.n(c),o=r(85893),u=a().bind(l());t.Z=function(e){var t=e.authors,r=e.publishedDate,i=(e.tags,r?s.ou.fromISO(r).toFormat("MMMM d, yyyy"):null);return(0,o.jsxs)("div",{className:u("container"),children:[t&&(0,o.jsxs)("span",{className:u("authors"),children:[(0,o.jsx)("span",{children:"By "}),t.map((function(e,r){return(0,o.jsxs)(n.Fragment,{children:[(0,o.jsx)("a",{href:"/blog?author=".concat(e.fields.slug),children:e.fields.name}),r<t.length-2&&(0,o.jsx)("span",{children:", "}),r===t.length-2&&(0,o.jsx)("span",{children:" and "})]},e.sys.id)})),i&&(0,o.jsx)("span",{children:"\xa0\xa0\xa0"})]}),(0,o.jsx)("span",{className:u("article-published-tags"),children:i&&(0,o.jsxs)("span",{className:u("article-published"),children:["|\xa0\xa0\xa0",i]})})]})}},67525:function(e,t,r){"use strict";r.d(t,{Z:function(){return Z}});r(67294);var n=r(11163),i=r(41664),a=r.n(i),s=r(47166),c=r.n(s),l=r(59499),o=r(99490),u=r(25948),d=r(58199),p=r(62062),f=r.n(p),h=r(85893);function m(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function g(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?m(Object(r),!0).forEach((function(t){(0,l.Z)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):m(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}r(9980)("zero");var _=c().bind(f()),y=function(e){var t=e.title,r=e.authors,n=e.publishedDate,i=e.intro,s=e.link,c=e.mainImage,l=e.mainImageFit,p=o.ou.fromISO(n).toFormat("MM . dd . yyyy");i&&i.length>250&&"".concat(i.slice(0,250).trim(),"...");return(0,h.jsxs)("div",{className:_("article"),children:[(0,h.jsx)(a(),{href:s,children:(0,h.jsx)("a",{className:_("article-image"),tabIndex:-1,children:(0,h.jsx)("div",{className:_("image-wrapper"),children:c&&(0,h.jsx)(d.Z,g(g({},c.fields),{},{contain:"cover"!==l,sizing:"w=760",description:null}))})})}),(0,h.jsxs)("div",{className:_("article-main"),children:[(0,h.jsx)("span",{className:_("published"),children:p}),(0,h.jsx)(a(),{href:s,children:(0,h.jsx)("a",{className:_("title"),children:(0,h.jsx)("h3",{children:t})})}),!1,(0,h.jsx)(u.Z,{authors:r})]})]})};y.defaultProps={authors:null,intro:null};var j=y,v=r(86862),b=r(27812),O=r(26593),x=r.n(O),w=r(80305),P=c().bind(x()),A=function(e){var t=e.types,r=void 0===t?[]:t,n=e.activeType,i=e.tags,a=void 0===i?[]:i,s=e.activeTag,c=e.onUpdate,l=[{name:"All Types",value:null}].concat((0,b.Z)(r.map((function(e){var t=e.fields;return{name:t.title,value:t.slug}})))),o=[{name:"All Products / Libraries",value:null}].concat((0,b.Z)(a.map((function(e){var t=e.fields;return{name:t.name,value:t.identifier}}))));return(0,h.jsx)("div",{className:P("root"),children:(0,h.jsxs)("div",{className:P("inner"),children:[(0,h.jsx)("div",{className:P("item"),children:(0,h.jsx)(w.Z,{label:"Types",selectedValue:n,options:l,onSelect:function(e){c({type:e,tag:s})},width:245})}),(0,h.jsx)("div",{className:P("item"),children:(0,h.jsx)(w.Z,{label:"Tags",selectedValue:s,options:o,onSelect:function(e){c({type:n,tag:e})},width:245})})]})})};A.defaultProps={};var N=A,I=c().bind(f()),D=function(e){var t=e.title,r=e.articles,i=e.author,a=e.types,s=e.tags,c=e.activeType,l=e.activeTag,o=e.page,u=e.totalPages,d=e.showTitle,p=void 0===d||d,f=e.onUpdateFilters,m=(0,n.useRouter)();return(0,h.jsx)("div",{className:I("container"),children:(0,h.jsxs)("div",{className:I("inner"),children:[p&&(0,h.jsxs)("div",{className:I("header"),children:[(0,h.jsx)("h1",{children:t}),(0,h.jsx)("div",{className:I("spacer")})]}),(0,h.jsx)(N,{types:a,activeType:c,tags:s,activeTag:l,onUpdate:function(e){var t=e.type,r=e.tag,n="/blog",a=o<2,s=!1;i&&(n="".concat(n,"?author=").concat(i.fields.slug),s=!0),t&&(n="".concat(n).concat(s?"&":"?","type=").concat(t),s=!0),r&&(n="".concat(n).concat(s?"&":"?","tag=").concat(r),s=!0),a?(m.push(n,void 0,{shallow:a}),f({type:t,tag:r,shallow:a})):window.location=n}}),(!r||r.length<1)&&(0,h.jsxs)("div",{className:I("empty"),children:[(0,h.jsx)("h2",{children:"No posts found."}),(0,h.jsx)("p",{children:(0,h.jsx)("a",{href:"/blog",children:"View all"})})]}),(0,h.jsx)("div",{className:I("list"),children:r.map((function(e){var t=e.fields,r=e.sys;return(0,h.jsx)(j,{title:t.title,intro:t.intro,authors:t.authors,publishedDate:t.publishedDate,mainImage:t.mainImage,mainImageFit:t.mainImageFit,link:"/blog/".concat(t.slug)},r.id)}))}),(0,h.jsx)(v.Z,{route:"/blog",author:i,page:o,type:c,tag:l,totalPages:u})]})})};D.defaultProps={articles:[],page:1,totalPages:1};var Z=D},28201:function(e,t,r){"use strict";r.d(t,{Z:function(){return P}});var n=r(59499),i=r(67294),a=r(47166),s=r.n(a),c=r(41664),l=r.n(c),o=r(99490),u=r(63112),d=r.n(u),p=r(58199),f=r(25948),h=r(85893);function m(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function g(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?m(Object(r),!0).forEach((function(t){(0,n.Z)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):m(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var _=r(9980)("zero"),y=s().bind(d()),j=function(e){var t=e.title,r=e.authors,n=e.publishedDate,i=(e.tags,e.intro),a=e.link,s=e.mainImage,c=e.mainImageFit,u=e.hideImage,d=o.ou.fromISO(n).toFormat("MM . dd . yyyy"),m=i&&i.length>250?"".concat(i.slice(0,250).trim(),"..."):i;return(0,h.jsxs)("div",{className:y("article",{"show-image":!u}),children:[!u&&(0,h.jsx)(l(),{href:a,children:(0,h.jsx)("a",{className:y("article-image"),tabIndex:-1,children:(0,h.jsx)("div",{className:y("image-wrapper"),children:s&&(0,h.jsx)(p.Z,g(g({},s.fields),{},{contain:"cover"!==c,sizing:"w=760",description:null}))})})}),(0,h.jsxs)("div",{className:y("article-main"),children:[(0,h.jsx)("span",{className:y("published"),children:d}),(0,h.jsx)(l(),{href:a,children:(0,h.jsx)("a",{className:y("title"),children:(0,h.jsx)("h3",{children:t})})}),!u&&i&&(0,h.jsx)("div",{dangerouslySetInnerHTML:{__html:_.render(m)}}),(0,h.jsx)(f.Z,{authors:r})]})]})};j.defaultProps={authors:null,tags:null,intro:null,mainImage:null,mainImageFit:"contain",hideImage:!1};var v=j;function b(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function O(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?b(Object(r),!0).forEach((function(t){(0,n.Z)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):b(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var x=s().bind(d()),w=function(e){var t=e.title,r=e.articles;if(r.length<3)return null;var n=r.slice(0,1),a=r.slice(1,3);return(0,h.jsx)("div",{className:x("container"),children:(0,h.jsxs)("div",{className:x("inner"),children:[(0,h.jsxs)("div",{className:x("header"),children:[(0,h.jsx)("h1",{children:t}),(0,h.jsx)("div",{className:x("spacer")})]}),(0,h.jsxs)("div",{className:x("list"),children:[n.map((function(e,t){var r=e.fields,n=e.sys;return(0,i.createElement)(v,O(O({},r),{},{link:"/blog/".concat(r.slug),key:n.id}))})),(0,h.jsx)("div",{className:x("list_inner"),children:a.map((function(e,t){var r=e.fields,n=e.sys;return(0,i.createElement)(v,O(O({},r),{},{link:"/blog/".concat(r.slug),key:n.id,hideImage:!0}))}))})]})]})})};w.defaultProps={articles:[]};var P=w},58199:function(e,t,r){"use strict";var n=r(92777),i=r(82262),a=r(45959),s=r(63553),c=r(37247),l=r(67294),o=r(47166),u=r.n(o),d=r(9980),p=r.n(d),f=r(11362),h=r.n(f),m=r(85893);function g(e){var t=function(){if("undefined"===typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"===typeof Proxy)return!0;try{return Boolean.prototype.valueOf.call(Reflect.construct(Boolean,[],(function(){}))),!0}catch(e){return!1}}();return function(){var r,n=(0,c.Z)(e);if(t){var i=(0,c.Z)(this).constructor;r=Reflect.construct(n,arguments,i)}else r=n.apply(this,arguments);return(0,s.Z)(this,r)}}var _=u().bind(h()),y=function(e){(0,a.Z)(r,e);var t=g(r);function r(e){var i;return(0,n.Z)(this,r),(i=t.call(this,e)).state={loaded:!1},i}return(0,i.Z)(r,[{key:"componentDidMount",value:function(){var e=this,t=this.props.file;if(t){var r=new Image;r.onload=function(){setTimeout((function(){e.setState({loaded:!0})}),0)},r.src=t.url}}},{key:"render",value:function(){var e=this.props,t=e.file,r=e.title,n=e.description,i=e.sizing,a=e.plain,s=e.contain,c=e.ratio,o=e.alignCaption,u=this.state.loaded,d=i?"".concat(t.url,"?").concat(i):t.url,f=c?{paddingBottom:"".concat(100*c,"%")}:{};return(0,m.jsxs)(l.Fragment,{children:[(0,m.jsx)("div",{className:_(["wrapper",{plain:a}]),style:f,children:(0,m.jsx)("div",{className:_(["image",{loaded:u,contain:s}]),children:(0,m.jsx)("img",{src:d||"",alt:r})})}),n&&(0,m.jsx)("span",{className:_("caption","align_".concat(o)),dangerouslySetInnerHTML:{__html:p()().renderInline(n)}})]})}}]),r}(l.PureComponent);y.defaultProps={file:{},title:"",description:null,sizing:null,plain:!1,contain:!1,alignCaption:""},t.Z=y},90606:function(e,t,r){"use strict";r.d(t,{H5:function(){return l},Wo:function(){return s},dz:function(){return o},pi:function(){return c}});var n=r(59499);function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function a(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){(0,n.Z)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var s=18,c=18,l=function(e){var t=e.page,r=e.type,n=e.allTypes,i=e.tag,c=e.allTags,l={skip:t>1?(t-2)*s+12:0,limit:1===t?12:s,order:"-fields.publishedDate,sys.id"};if(r){var o=n.filter((function(e){return e.fields.slug===r}))[0];l=a(a({},l),{},{"fields.type.sys.id":o.sys.id})}if(i){var u=c.filter((function(e){return e.fields.identifier===i}))[0];l=a(a({},l),{},{"fields.tags.sys.id":u.sys.id})}return l},o=function(e){var t=e.page,r=e.type,n=e.allTypes,i=e.tag,s=e.allTags,l={skip:t>1?(t-2)*c+12:0,limit:1===t?12:c,order:"-fields.startDate,sys.id"};if(r){var o=n.filter((function(e){return e.fields.slug===r}))[0];l=a(a({},l),{},{"fields.category.sys.id":o.sys.id})}if(i){var u=s.filter((function(e){return e.fields.identifier===i}))[0];l=a(a({},l),{},{"fields.tags.sys.id":u.sys.id})}return l}},25949:function(e){e.exports={authors:"ArticleDetails_authors__aqDWy","article-published-tags":"ArticleDetails_article-published-tags__9VxRY"}},62062:function(e){e.exports={container:"ArticlesList_container__mBEpW",inner:"ArticlesList_inner__QWc69",header:"ArticlesList_header__45BKa",spacer:"ArticlesList_spacer__8l_nL",list:"ArticlesList_list__uP0RC",article:"ArticlesList_article__D2wUc","article-image":"ArticlesList_article-image__C8mJ5","image-wrapper":"ArticlesList_image-wrapper__6pP_O","article-main":"ArticlesList_article-main__umhHy",published:"ArticlesList_published__IjEQt",title:"ArticlesList_title__h4t7k",tag:"ArticlesList_tag__ifrv3",tags:"ArticlesList_tags__DtXAu"}},26593:function(e){e.exports={root:"BlogFilters_root__mrUMs",inner:"BlogFilters_inner__87PZK",item:"BlogFilters_item__1GpID"}},63112:function(e){e.exports={container:"FeaturedArticles_container__qjdPd",inner:"FeaturedArticles_inner__TJ_IO",header:"FeaturedArticles_header__HH_4Y",spacer:"FeaturedArticles_spacer__s9fFO",list:"FeaturedArticles_list__hYg7Q",article:"FeaturedArticles_article___xsNU","article-image":"FeaturedArticles_article-image__P3efR","image-wrapper":"FeaturedArticles_image-wrapper__tAK_P","show-image":"FeaturedArticles_show-image___K_J4","article-main":"FeaturedArticles_article-main__C8qKx",published:"FeaturedArticles_published__FBcGT",title:"FeaturedArticles_title__mKuQc"}},11362:function(e){e.exports={wrapper:"Image_wrapper__beWl0",plain:"Image_plain__m5QAa",image:"Image_image__EAuYA",contain:"Image_contain__sI5D9",loaded:"Image_loaded___CdvS",caption:"Image_caption__outy2",align_left:"Image_align_left__VA3hB"}},52587:function(e,t,r){"use strict";function n(e,t){(null==t||t>e.length)&&(t=e.length);for(var r=0,n=new Array(t);r<t;r++)n[r]=e[r];return n}r.d(t,{Z:function(){return n}})},27812:function(e,t,r){"use strict";r.d(t,{Z:function(){return a}});var n=r(52587);var i=r(2937);function a(e){return function(e){if(Array.isArray(e))return(0,n.Z)(e)}(e)||function(e){if("undefined"!==typeof Symbol&&null!=e[Symbol.iterator]||null!=e["@@iterator"])return Array.from(e)}(e)||(0,i.Z)(e)||function(){throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}},2937:function(e,t,r){"use strict";r.d(t,{Z:function(){return i}});var n=r(52587);function i(e,t){if(e){if("string"===typeof e)return(0,n.Z)(e,t);var r=Object.prototype.toString.call(e).slice(8,-1);return"Object"===r&&e.constructor&&(r=e.constructor.name),"Map"===r||"Set"===r?Array.from(e):"Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)?(0,n.Z)(e,t):void 0}}}}]);