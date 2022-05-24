#!/usr/bin/env/python

import tarfile
import json

class VisualizationHtml(object):

    def __init__(self, input_text, indx, dist, image_data_path=None):
        if image_data_path is None:
            print(f"image_data_path must be specified.")
            raise ValueError
        self.idp = image_data_path
        self.html = "<!DOCTYPE html><html class='fullscreen_page sticky_menu'><head><meta http-equiv='Content-Type' content='text/html; charset=UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1, maximum-scale=1'><title>Retrieval Visualization</title><link href='http://fonts.googleapis.com/css?family=PT+Sans' rel='stylesheet' type='text/css'><link href='http://fonts.googleapis.com/css?family=Roboto:400,300,500,900' rel='stylesheet' type='text/css'><link rel='stylesheet' href='css/theme.css' type='text/css' media='all' /><link rel='stylesheet' href='css/responsive.css' type='text/css' media='all' /><link rel='stylesheet' href='css/custom.css' type='text/css' media='all' /><script type='text/javascript' src='js/jquery.min.js'></script></head><body>"
        self.footer = "   </div></div><div class='preloader'></div> <footer class='fullwidth'><div class='footer_wrapper'><div class='copyright'>Image Retrieval Visualization. Copyright 2021-2022 &copy; Haoyu Song. All Rights Reserved.</div>    </div></footer>    	    <div class='content_bg'></div><script type='text/javascript' src='js/jquery-ui.min.js'></script> <script type='text/javascript' src='js/modules.js'></script><script type='text/javascript' src='js/theme.js'></script> </body></html>"
        self.header = self.set_header(input_text)
        self.body = self.set_body(indx, dist)


    def set_header(self, input_text):
        return f"<header class='main_header'><div class='header_wrapper'><div class='logo_sect'><a href='index.html' class='logo'><img src='img/text.png' alt=''class='logo_def'></a><div class='slogan'><div class='blogpost_fw_content'>{input_text}</div></div></div> </header> <div class='fullscreen_block'><div class='fs_blog_module is_masonry this_is_blog'>"

    def set_body(self, indx, dist):
        assert  len(indx) == len(dist)
        body = ""
        rank = 0
        for i, dist in zip(indx, dist):
            name = str(i).zfill(9)
            tarfile_name = f"{str(i//10000).zfill(5)}.tar"
            tar_path = f"{self.idp}/{tarfile_name}"
            with tarfile.open(tar_path, "r") as file:
                for i_n in file.getmembers():
                    if f"{name}.json" in i_n.name:
                        meta = json.load(file.extractfile(i_n))
                        caption = meta["caption"]
                        url = meta["url"]
                        break
            rank += 1
            body += f"<div class='blogpost_preview_fw'><div class='fw_preview_wrapper'><div class='pf_output_container'><img class='featured_image_standalone' src='{url}' alt='' /></div><div class='blogpreview_top'><div class='box_date'><span class='box_month'>{rank}</span></div><div class='listing_meta'><span>Distance</a></span><span>{dist}</a></span></div></div><h6 class='blogpost_title'>{name}.jpg</a></h6><div class='blogpost_fw_content'><article class='contentarea'>{caption}</article></div></div></div>"
        return body

    def save_report(self, tgt=None):
        assert tgt is not None
        with open(tgt,'w') as t_file:
            t_file.write(f"{self.html}{self.header}{self.body}{self.footer}")
        print(f"Reports have been written at {tgt}")