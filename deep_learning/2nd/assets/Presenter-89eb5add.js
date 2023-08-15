import{o as d,f as k,g as e,B as y,C as D,v as h,d as M,i as B,a as P,D as S,x as v,E as H,_ as z,G as I,H as R,c as b,b as j,I as $,J as A,K as E,L,M as q,N as F,O,P as U,Q as W,h as i,m as u,t as Z,n as x,R as N,S as V,p as G,T as J,U as w,V as K,F as Q,W as X,X as Y,w as ee,Y as te,Z as se,q as T,$ as oe,a0 as ne,a1 as le,a2 as ae,k as ie,a3 as re,a4 as ce}from"./index-3813cbd4.js";import{N as ue}from"./NoteDisplay-737b538c.js";import de from"./DrawingControls-e8d5710e.js";const _e={class:"slidev-icon",viewBox:"0 0 32 32",width:"1.2em",height:"1.2em"},pe=e("path",{fill:"currentColor",d:"M12 10H6.78A11 11 0 0 1 27 16h2A13 13 0 0 0 6 7.68V4H4v8h8zm8 12h5.22A11 11 0 0 1 5 16H3a13 13 0 0 0 23 8.32V28h2v-8h-8z"},null,-1),me=[pe];function ve(o,l){return d(),k("svg",_e,me)}const he={name:"carbon-renew",render:ve},fe={class:"slidev-icon",viewBox:"0 0 32 32",width:"1.2em",height:"1.2em"},ge=e("path",{fill:"currentColor",d:"M16 30a14 14 0 1 1 14-14a14 14 0 0 1-14 14Zm0-26a12 12 0 1 0 12 12A12 12 0 0 0 16 4Z"},null,-1),xe=e("path",{fill:"currentColor",d:"M20.59 22L15 16.41V7h2v8.58l5 5.01L20.59 22z"},null,-1),we=[ge,xe];function ye(o,l){return d(),k("svg",fe,we)}const Se={name:"carbon-time",render:ye},ke="/orientation/deep_learning/2nd/assets/logo-title-horizontal-96c3c915.png";function Ce(){const o=y(Date.now()),l=D({interval:1e3}),_=h(()=>{const t=(l.value-o.value)/1e3,n=Math.floor(t%60).toString().padStart(2,"0");return`${Math.floor(t/60).toString().padStart(2,"0")}:${n}`});function p(){o.value=l.value}return{timer:_,resetTimer:p}}const be=M({__name:"NoteStatic",props:{class:{type:String,required:!1}},setup(o){const l=o;B(P);const _=h(()=>{var t,n,s;return(s=(n=(t=S.value)==null?void 0:t.meta)==null?void 0:n.slide)==null?void 0:s.note}),p=h(()=>{var t,n,s;return(s=(n=(t=S.value)==null?void 0:t.meta)==null?void 0:n.slide)==null?void 0:s.noteHTML});return(t,n)=>(d(),v(ue,{class:H(l.class),note:_.value,"note-html":p.value},null,8,["class","note","note-html"]))}}),$e=z(be,[["__file","/home/toonnyy8/projects/orientation/deep_learning/node_modules/@slidev/client/internals/NoteStatic.vue"]]),f=o=>(X("data-v-574fd206"),o=o(),Y(),o),Ne={class:"bg-main h-full slidev-presenter"},Ve={class:"grid-container"},Te={class:"grid-section top flex"},Me=f(()=>e("img",{src:ke,class:"ml-2 my-auto h-10 py-1 lg:h-14 lg:py-2",style:{height:"3.5rem"}},null,-1)),Be=f(()=>e("div",{class:"flex-auto"},null,-1)),Pe={class:"text-2xl pl-2 pr-6 my-auto tabular-nums"},He=f(()=>e("div",{class:"context"}," current ",-1)),ze=f(()=>e("div",{class:"context"}," next ",-1)),De={class:"grid-section note overflow-auto"},Ie={class:"grid-section bottom"},Re={class:"progress-bar"},je=M({__name:"Presenter",setup(o){B(P);const l=y();I(),R(l);const _=b.titleTemplate.replace("%s",b.title||"Slidev");j({title:`Presenter - ${_}`});const{timer:p,resetTimer:t}=Ce(),n=y([]),s=h(()=>$.value<A.value?{route:S.value,clicks:$.value+1}:E.value?{route:L.value,clicks:0}:null);return q(()=>{const C=l.value.querySelector("#slide-content"),r=F(O()),g=U();W(()=>{if(!g.value||te.value||!se.value)return;const c=C.getBoundingClientRect(),a=(r.x-c.left)/c.width*100,m=(r.y-c.top)/c.height*100;if(!(a<0||a>100||m<0||m>100))return{x:a,y:m}},c=>{ee.cursor=c})}),(C,r)=>{const g=Se,c=he;return d(),k(Q,null,[e("div",Ne,[e("div",Ve,[e("div",Te,[Me,Be,e("div",{class:"timer-btn my-auto relative w-22px h-22px cursor-pointer text-lg",opacity:"50 hover:100",onClick:r[0]||(r[0]=(...a)=>i(t)&&i(t)(...a))},[u(g,{class:"absolute"}),u(c,{class:"absolute opacity-0"})]),e("div",Pe,Z(i(p)),1)]),e("div",{ref_key:"main",ref:l,class:"relative grid-section main flex flex-col p-2 lg:p-4",style:x(i(T))},[u(V,{key:"main",class:"h-full w-full"},{default:N(()=>[u(oe,{context:"presenter"})]),_:1}),He],4),e("div",{class:"relative grid-section next flex flex-col p-2 lg:p-4",style:x(i(T))},[s.value?(d(),v(V,{key:"next",class:"h-full w-full"},{default:N(()=>{var a;return[u(i(le),{is:(a=s.value.route)==null?void 0:a.component,"clicks-elements":n.value,"onUpdate:clicksElements":r[1]||(r[1]=m=>n.value=m),clicks:s.value.clicks,"clicks-disabled":!1,class:H(i(ne)(s.value.route)),route:s.value.route,context:"previewNext"},null,8,["is","clicks-elements","clicks","class","route"])]}),_:1})):G("v-if",!0),ze],4),e("div",De,[(d(),v($e,{key:1,class:"w-full max-w-full h-full overflow-auto p-2 lg:p-4"}))]),e("div",Ie,[u(re,{persist:!0})]),(d(),v(de,{key:0}))]),e("div",Re,[e("div",{class:"progress h-2px bg-primary transition-all",style:x({width:`${(i(ae)-1)/(i(ie)-1)*100}%`})},null,4)])]),u(ce),u(K,{modelValue:i(w),"onUpdate:modelValue":r[2]||(r[2]=a=>J(w)?w.value=a:null)},null,8,["modelValue"])],64)}}});const qe=z(je,[["__scopeId","data-v-574fd206"],["__file","/home/toonnyy8/projects/orientation/deep_learning/node_modules/@slidev/client/internals/Presenter.vue"]]);export{qe as default};
