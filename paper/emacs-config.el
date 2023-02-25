;;; emacs-configinit.el --- Minimal emacs config for exporting paper.org to paper.pdf -*- lexical-binding: t; -*-
;;
;; Copyright (C) 2022 Aidan Scannell
;;
;; Author: Aidan Scannell <scannell.aidan@gmail.com>
;; Maintainer: Aidan Scannell <scannell.aidan@gmail.com>
;; Created: August 18, 2022
;; Modified: August 18, 2022
;; Version: 0.0.1
;; Package-Requires: ((emacs "24.3"))
;;
;; This file is not part of GNU Emacs.
;;
;;; Commentary:
;;
;;  Description
;;
;;; Code:
(require 'package)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)
(package-initialize) ;; Initialize & Install Package

(unless (package-installed-p 'use-package)
  (package-refresh-contents)
  (package-install 'use-package))
(eval-and-compile
  (setq use-package-always-ensure t
        use-package-expand-minimally t))


;; (setq org-directory (concat (getenv "HOME")"/Library/Mobile Documents/com~apple~CloudDocs/org"))
;; (setq zot_bib (concat org-directory "/ref/zotero-library.bib"))
(use-package citar
  :bind (("C-c b" . citar-insert-citation)
         :map minibuffer-local-map
         ("M-b" . citar-insert-preset)))
  ;; :custom
  ;; (citar-bibliography '(zot_bib,"zotero-library.bib")))
  ;; (citar-bibliography '("../zotero-library.bib")))

(use-package org
  :config

  (use-package org-ref)

  (use-package org-contrib
    :config
    (require 'ox-extra)
    (ox-extras-activate '(latex-header-blocks ignore-headlines)))

  (setq org-latex-pdf-process
        '("mkdir build \n latexmk -cd  -f -interaction=nonstopmode -outdir=build -auxdir=build -output-format=pdf %f"))

  (unless (boundp 'org-latex-classes)
    (setq org-latex-classes nil))

  ;; TODO remove this when not submitting as blind for reviewing
  (setq org-latex-with-hyperref nil)

  (add-to-list 'org-latex-classes
               '("two-side-article"
                 "\\documentclass[twoside]{article}
    [NO-DEFAULT-PACKAGES]
    [PACKAGES]
    [EXTRA]
    \\newcommand{\\mboxparagraph}[1]{\\paragraph{#1}\\mbox{}\\\\}
    \\newcommand{\\mboxsubparagraph}[1]{\\subparagraph{#1}\\mbox{}\\\\}"
                 ("\\section{%s}" . "\\section*{%s}")
                 ("\\subsection{%s}" . "\\subsection*{%s}")
                 ("\\subsubsection{%s}" . "\\subsubsection*{%s}")
                 ("\\paragraph{%s}" . "\\paragraph*{%s}"))))

;; (setq org-latex-hyperref-template
;;       "\\\hypersetup{
;;     pdfauthor={%a},
;;     pdftitle={%t},
;;     pdfkeywords={%k},
;;     pdfsubject={%d},
;;     pdfcreator={%c},
;;     pdflang={%L}}\n")


(provide 'init)
;;; emacs-config.el ends here
