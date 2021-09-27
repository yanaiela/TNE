	document.getElementById("page").innerHTML = '    <div class = "container" id = "top_l">          <article class="fragment" id="fragm">             <div id=\'text\'></div>          </article>       </div>       <div class = "container" id = "top_r">                   <div id="items">              <p id = "np_title" style="margin:2;margin-bottom:10;margin-top:11;"><b>NPs:</b></p>             <div id="item_pane"></div>          </div>       </div>       <div class = "container" id = "mid_r">          <div id="entities">             <p style="margin:2;margin-top:6px;"></p>             <div id="entity_pane">             </div>          </div>       </div>       <div class = "container" id = "mid_l">                   <div id="options_pane">             <div id=\'options\'></div>          </div>       </div>       <div class = "container" id = "bottom_l">          <div id="button">          </div>       </div>';

	var _text = new Ractive({
		target: '#text',
		template: '{{{ render_text(nps, text, entities, current, bridge_candidates) }}}',
		data: {
			nps: nps,
			text: text,
			current: 0,
			url:url,
			entities: coref,
			render_text: f_render_text,
			bridge_candidates: [],
		},
	});

	var _entity_pane = new Ractive({
		target: '#entity_pane',
		template: '<p id = "rel_title" style="margin:12.5;margin-left:5"><b>NP Relations:</b> </p>             <ol>{{#each links_to_present(bridging_results,current) as link}}       <li style="margin:2;margin-left:20;">{{text_from_id(link.bridge)}} [{{link.preposition}} {{#each get_complement(link.complement).members as member}} {{#if (!pronouns.includes(member))}}           <span style="background-color:{{entity_color(get_complement(link.complement))}};" on-mouseover="[\'highlight\', (get_complement(link.complement)), entity_color(get_complement(link.complement))]" on-mouseout="[\'highlight\', (get_complement(link.complement)), \'transparent\']">                 {{text_from_id(member)}}             </span> {{space}} {{/if}} {{/each}}]                 </li>       {{/each}}       </ol>             <div id = "lnks">             <ol>{{#each merged_bridges as link}}       <li style="margin:2;margin-left:20;">{{text_from_id(link.bridge)}} [{{link.preposition}}  {{#each get_complement(link.complement).members as member}} {{#if (!pronouns.includes(member))}}           <span style="background-color:{{entity_color(get_complement(link.complement))}};" on-mouseover="[\'highlight\', (get_complement(link.complement)), entity_color(get_complement(link.complement))]" on-mouseout="[\'highlight\', (get_complement(link.complement)), \'transparent\']">                 {{text_from_id(member)}}             </span> {{space}} {{/if}} {{/each}}]               </li>       {{/each}}       </ol>       </div>       ',
		data: {
			bridging_results: BRIDGING_RESULTS,
			current: 0,
			bridges: bridges,
			entities: coref,
			nps: nps,
			text_from_id: f_np_text_from_id,
			hidden: true,
			space: " ",
			pronouns: [],
			links_to_present: function(result, current) {

				var values = [];
				var keys = Object.keys(result);
				keys.forEach(function(key) {
					values.push(result[key]);
				});
				console.log(values.filter(lnk => lnk.bridge == current));
				return (values.filter(lnk => lnk.bridge == current))
			},

			get_complement: function(entity_id) {
				return (this.get("entities").filter(e => e.id == entity_id)[0])
			},
			entity_color: function(entity) {
				return (COLORS[entity.id % COLORS.length])
			},
		},
		computed: {
			merged_bridges: function() {
				var mb = [];
				this.get('bridges').forEach(lnk => {
					var bridge_text = this.get("nps")[lnk.bridge].text;
					var bridge_entity = this.get("entities").filter(e => e.members.includes(lnk.bridge))[0].id;
					if (mb.filter(l => this.get("nps")[l.bridge].text == bridge_text && this.get("entities").filter(e => e.members.includes(l.bridge))[0].id == bridge_entity && l.complement == lnk.complement).length == 0) {
						mb.push(lnk);
					}
				});
				return mb;
			},
		}

	});

	var entity_listener = _entity_pane.on('highlight', function(ctx, entity, color) {
		highlightClass("entity" + entity.id, color);
	});

	var _item_pane = new Ractive({
		target: '#item_pane',
		template: '{{#each nps as np}}       {{#if np.id == current_np_id}}             <span class = "item" id = "current_item" style = "background-color:lightyellow" on-click="[\'go_to_np\', np]"> {{np.text}}</span>       {{elseif done_items[np.id]}}             <span class = "item" style = "color:gray" on-click="[\'go_to_np\', np]"><s> {{np.text}}</s></span>       {{else}}             <span class = "item" on-click="[\'go_to_np\', np]"> {{np.text}}</span>       {{/if}}       {{/each}}',
		data: {
			nps: nps,
			current_np_id: 0,
		},
	});

	var item_listener = _item_pane.on('go_to_np', function(ctx, np) {
		go_to_np(np.id);
	});

	var _options = new Ractive({
		target: '#options',
		template: '<p id = "coref_title" style="margin:2;margin-left:5;"><b>Coreference Chains:</b> </p>       <ol>           {{#each entities as entity}}           <li style="margin:2;margin-left:-10">               {{#each entity.members as member}} {{#if (!pronouns.includes(member))}}       <span style="background-color:{{entity_color(entity)}};" on-mouseover="[\'highlight\', entity, entity_color(entity)]" on-mouseout="[\'highlight\', entity, \'transparent\']">                 {{text_from_id(member)}}             </span> {{space}} {{/if}} {{/each}}          </li>           {{/each}}       </ol>',
		data: {

			entities: coref,
			pronouns: [],
			text_from_id: f_np_text_from_id,
			space: " ",
			entity_color: function(entity) {
				return (COLORS[entity.id % COLORS.length])
			},
		},
	});

	var option_listener = _options.on('highlight', function(ctx, entity, color) {
		highlightClass("entity" + entity.id, color);
	});



	var _button = new Ractive({
		target: '#button',
		template: '<input twoway=\'false\' type="button" class = "button btn btn-primary" id="prev" disabled = {{idx == 1 && hit_id == hit_ids[0]}} value="Previous" on-click="[\'display_prev_idx\', idx, hit_id, hit_ids]"> 	  <input type="text" id="indexbox"  value={{idx}} on-change="[\'display_idx\', idx, hit_id, hit_ids]">       <p id = "range"><i>(out of {{hit_ids.length}})</i></p>	 	        <input  style = "margin-top:7.5;float:right;" twoway=\'false\' type="button" class = "button btn btn-primary" id="next" disabled = {{idx == hit_ids.length && hit_id == hit_ids[hit_ids.length - 1]}} value="Next" on-click="[\'display_next_idx\', idx, hit_id, hit_ids]">',
		data: {
			hit_id: HIT_ID,
			idx: IDX,
			hit_ids: CURR_SPLIT_HIT_IDS,
		}
	});



	_button.on('display_prev_idx', function(ctx, idx, hit_id, hit_ids) {
		prev_idx = idx * 1 - 1;		
		new_url = document.URL.replace(hit_id, hit_ids[prev_idx - 1]);
		if (hit_ids.length >= prev_idx && prev_idx > 0) {
			window.location.href = new_url;
		} else {
			return;
		}
	});

	_button.on('display_next_idx', function(ctx, idx, hit_id, hit_ids) {
		next_idx = idx * 1 + 1;
		console.log(next_idx);
		console.log(hit_ids[next_idx - 1]);
		new_url = document.URL.replace(hit_id, hit_ids[next_idx - 1]);
		if (hit_ids.length >= next_idx && next_idx > 0) {
			window.location.href = new_url;
		} else {
			return;
		}
	});

	_button.on('display_idx', function(ctx, idx, hit_id, hit_ids) {
		if (hit_ids.length >= idx && idx > 0) {
			new_url = document.URL.replace(hit_id, hit_ids[idx - 1]);
			window.location.href = new_url;
		} else {
			document.getElementById("indexbox").style.backgroundColor = "red";
			this.set({
				'idx': IDX,
			});
			setTimeout(function() {
				document.getElementById("indexbox").style.backgroundColor = ""
			}, 1000);
		}
	});


	get_bridging_results();