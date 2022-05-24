"use strict";

var header = jQuery('.main_header'),
    html = jQuery('html'),
    body = jQuery('body'),
    footer = jQuery('footer'),
    window_h = jQuery(window).height(),
    window_w = jQuery(window).width(),
    main_wrapper = jQuery('.main_wrapper'),
    site_wrapper = jQuery('.site_wrapper'),
    preloader_block = jQuery('.preloader'),
    fullscreen_block = jQuery('.fullscreen_block'),
    is_masonry = jQuery('.is_masonry'),
    grid_portfolio_item = jQuery('.grid-portfolio-item'),
    pp_block = jQuery('.pp_block'),
    head_border = 1;

jQuery(document).ready(function ($) {
	"use strict";
    if (jQuery('.preloader').size() > 0) {
        setTimeout("preloader_block.addClass('la-animate');", 500);
        setTimeout("preloader_block.addClass('load_done')", 2500);
        setTimeout("preloader_block.remove()", 2950);
    }
	if (html.hasClass('sticky_menu') && body.hasClass('admin-bar')) {
		header.css('top', jQuery('#wpadminbar').height());
	}
    content_update();
    if (jQuery('.flickr_widget_wrapper').size() > 0) {
        jQuery('.flickr_badge_image a').each(function () {
            jQuery(this).append('<div class="flickr_fadder"></div>');
        });
    }
    header.find('.header_wrapper').append('<a href="javascript:void(0)" class="menu_toggler"></a>');
    header.append('<div class="mobile_menu_wrapper"><ul class="mobile_menu container"/></div>');
    jQuery('.mobile_menu').html(header.find('.menu').html());
    jQuery('.mobile_menu_wrapper').hide();
    jQuery('.menu_toggler').click(function () {
        jQuery('.mobile_menu_wrapper').slideToggle(300);
        jQuery('.main_header').toggleClass('opened');
    });
    setTimeout("jQuery('body').animate({'opacity' : '1'}, 500)", 500);

    jQuery('.search_toggler').click(function (event) {
		event.preventDefault();
        header.toggleClass('search_on');
    });
    if (pp_block.size()) {
        centerWindow404();
    }
	
	// prettyPhoto
	jQuery("a[rel^='prettyPhoto'], .prettyPhoto").prettyPhoto();	
	
	jQuery('a[data-rel]').each(function() {
		$(this).attr('rel', $(this).data('rel'));
	});
	
	/* NivoSlider */
	jQuery('.nivoSlider').each(function(){
		jQuery(this).nivoSlider({
			directionNav: false,
			controlNav: true,
			effect:'fade',
			pauseTime:4000,
			slices: 1
		});
	});
	
	/* Accordion & toggle */
	jQuery('.shortcode_accordion_item_title').click(function(){
		if (!jQuery(this).hasClass('state-active')) {
			jQuery(this).parents('.shortcode_accordion_shortcode').find('.shortcode_accordion_item_body').slideUp('fast',function(){
				content_update();
			});
			jQuery(this).next().slideToggle('fast',function(){
				content_update();
			});
			jQuery(this).parents('.shortcode_accordion_shortcode').find('.state-active').removeClass('state-active');
			jQuery(this).addClass('state-active');
		}
	});
	jQuery('.shortcode_toggles_item_title').click(function(){
		jQuery(this).next().slideToggle('fast',function(){
			content_update();
		});
		jQuery(this).toggleClass('state-active');
	});

	jQuery('.shortcode_accordion_item_title.expanded_yes, .shortcode_toggles_item_title.expanded_yes').each(function( index ) {
		jQuery(this).next().slideDown('fast',function(){
			content_update();
		});
		jQuery(this).addClass('state-active');
	});
	
	/* Counter */
	if (jQuery(window).width() > 760) {						
		jQuery('.shortcode_counter').each(function(){							
			if (jQuery(this).offset().top < jQuery(window).height()) {
				if (!jQuery(this).hasClass('done')) {
					var set_count = jQuery(this).find('.stat_count').attr('data-count');
					jQuery(this).find('.stat_temp').stop().animate({width: set_count}, {duration: 3000, step: function(now) {
							var data = Math.floor(now);
							jQuery(this).parents('.counter_wrapper').find('.stat_count').html(data);
						}
					});	
					jQuery(this).addClass('done');
					jQuery(this).find('.stat_count');
				}							
			} else {
				jQuery(this).waypoint(function(){
					if (!jQuery(this).hasClass('done')) {
						var set_count = jQuery(this).find('.stat_count').attr('data-count');
						jQuery(this).find('.stat_temp').stop().animate({width: set_count}, {duration: 3000, step: function(now) {
								var data = Math.floor(now);
								jQuery(this).parents('.counter_wrapper').find('.stat_count').html(data);
							}
						});	
						jQuery(this).addClass('done');
						jQuery(this).find('.stat_count');
					}
				},{offset: 'bottom-in-view'});								
			}														
		});
	} else {
		jQuery('.shortcode_counter').each(function(){							
			var set_count = jQuery(this).find('.stat_count').attr('data-count');
			jQuery(this).find('.stat_temp').animate({width: set_count}, {duration: 3000, step: function(now) {
					var data = Math.floor(now);
					jQuery(this).parents('.counter_wrapper').find('.stat_count').html(data);
				}
			});
			jQuery(this).find('.stat_count');
		},{offset: 'bottom-in-view'});	
	}
	
	/* Tabs */
	jQuery('.shortcode_tabs').each(function(index) {
		/* GET ALL HEADERS */
		var i = 1;
		jQuery(this).find('.shortcode_tab_item_title').each(function(index) {
			jQuery(this).addClass('it'+i); jQuery(this).attr('whatopen', 'body'+i);
			jQuery(this).addClass('head'+i);
			jQuery(this).parents('.shortcode_tabs').find('.all_heads_cont').append(this);
			i++;
		});
	
		/* GET ALL BODY */
		var i = 1;
		jQuery(this).find('.shortcode_tab_item_body').each(function(index) {
			jQuery(this).addClass('body'+i);
			jQuery(this).addClass('it'+i);
			jQuery(this).parents('.shortcode_tabs').find('.all_body_cont').append(this);
			i++;
		});
	
		/* OPEN ON START */
		jQuery(this).find('.expand_yes').addClass('active');
		var whatopenOnStart = jQuery(this).find('.expand_yes').attr('whatopen');
		jQuery(this).find('.'+whatopenOnStart).addClass('active');
	});
	
	jQuery(document).on('click', '.shortcode_tab_item_title', function(){
		jQuery(this).parents('.shortcode_tabs').find('.shortcode_tab_item_body').removeClass('active');
		jQuery(this).parents('.shortcode_tabs').find('.shortcode_tab_item_title').removeClass('active');
		var whatopen = jQuery(this).attr('whatopen');
		jQuery(this).parents('.shortcode_tabs').find('.'+whatopen).addClass('active');
		jQuery(this).addClass('active');
		content_update();
	});
	
	/* Messagebox */
	jQuery('.shortcode_messagebox').find('.box_close').click(function(){
		jQuery(this).parents('.module_messageboxes').fadeOut(400);
	});
	
	/* Skills */
	jQuery('.chart').each(function(){
		jQuery(this).css({'font-size' : jQuery(this).parents('.skills_list').attr('data-fontsize'), 'line-height' : jQuery(this).parents('.skills_list').attr('data-size')});
		jQuery(this).find('span').css('font-size' , jQuery(this).parents('.skills_list').attr('data-fontsize'));
	});

	if (jQuery(window).width() > 760) {
		jQuery('.skill_li').waypoint(function(){
			jQuery('.chart').each(function(){
				jQuery(this).easyPieChart({
					barColor: jQuery(this).parents('ul.skills_list').attr('data-color'),
					trackColor: jQuery(this).parents('ul.skills_list').attr('data-bg'),
					scaleColor: false,
					lineCap: 'square',
					lineWidth: parseInt(jQuery(this).parents('ul.skills_list').attr('data-width')),
					size: parseInt(jQuery(this).parents('ul.skills_list').attr('data-size')),
					animate: 1500
				});
			});
		},{offset: 'bottom-in-view'});
	} else {
		jQuery('.chart').each(function(){
			jQuery(this).easyPieChart({
				barColor: jQuery(this).parents('ul.skills_list').attr('data-color'),
				trackColor: jQuery(this).parents('ul.skills_list').attr('data-bg'),
				scaleColor: false,
				lineCap: 'square',
				lineWidth: parseInt(jQuery(this).parents('ul.skills_list').attr('data-width')),
				size: parseInt(jQuery(this).parents('ul.skills_list').attr('data-size')),
				animate: 1500
			});
		});
	}
	
	// contact form
	jQuery("#ajax-contact-form").submit(function() {
		var str = $(this).serialize();		
		$.ajax({
			type: "POST",
			url: "contact_form/contact_process.php",
			data: str,
			success: function(msg) {
				// Message Sent - Show the 'Thank You' message and hide the form
				if(msg == 'OK') {
					var result = '<div class="notification_ok">Your message has been sent. Thank you!</div>';
					jQuery("#fields").hide();
				} else {
					var result = msg;
				}
				jQuery('#note').html(result);
			}
		});
		return false;
	});
		
});

jQuery(window).resize(function () {
	"use strict";
    window_h = jQuery(window).height();
    window_w = jQuery(window).width();
    content_update();
});

jQuery(window).load(function () {
	"use strict";
    content_update();
});

function content_update() {
	"use strict";
    if (html.hasClass('sticky_menu')) {
        if (html.hasClass('without_border')) {
            head_border = 0;
        }
        jQuery('body').css('padding-top', header.height() + head_border);
    }
    site_wrapper.width('100%').css('min-height', window_h - parseInt(body.css('padding-top')));
    if (jQuery(window).width() > 760) {
        main_wrapper.css('min-height', window_h - header.height() - footer.height() - parseInt(main_wrapper.css('padding-top')) - parseInt(main_wrapper.css('padding-bottom')) - parseInt(footer.css('border-top-width')) - parseInt(header.css('border-bottom-width')) + 'px');
        if (fullscreen_block.size() > 0 && footer.size() > 0) {
            fullscreen_block.css('min-height', window_h - header.height() - footer.height() - parseInt(fullscreen_block.css('padding-top')) - parseInt(fullscreen_block.css('padding-bottom')) - parseInt(footer.css('border-top-width')) - parseInt(header.css('border-bottom-width')) + 'px');
        } else {
            fullscreen_block.css('min-height', window_h - header.height() - parseInt(header.css('border-bottom-width')) + 'px');
        }
    } else {
        //
    }
}

function animateList() {
	"use strict";
    jQuery('.loading:first').removeClass('loading').animate({'z-index': '15'}, 200, function () {
        animateList();
        if (is_masonry.size() > 0) {
            is_masonry.masonry();
        }
    });
};

var setTop = 0;
function centerWindow404() {
	"use strict";
    setTop = (window_h - pp_block.height() - header.height()) / 2 + header.height();
    if (setTop < header.height() + 50) {
        pp_block.addClass('fixed');
        body.addClass('addPadding');
        pp_block.css('top', header.height() + 50 + 'px');
    } else {
        pp_block.css('top', setTop + 'px');
        pp_block.removeClass('fixed');
        body.removeClass('addPadding');
    }
}

jQuery(window).resize(function () {
	"use strict";
    if (pp_block.size()) {
        setTimeout('centerWindow404()', 500);
        setTimeout('centerWindow404()', 1000);
    }
});